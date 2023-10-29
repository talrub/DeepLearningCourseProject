import jax
import numpy
from jax import vmap, random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from matplotlib import pyplot as plt
from PIL import Image
import math

def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)


def relu_layer(params, x):
    """ Simple ReLu layer for single sample """
    return ReLU(jnp.matmul(params[0], x) + params[1])

def encoder_forward_pass(params, embeddings_type, input_array): # input_array.shape = (num_of_examples,1,784,1)
    if embeddings_type == "pix_to_vec_to_vec":
        W1, b1, W2, b2 = params # W1.shape=(1,model_count,1,H_in) W2.shape=(1,model_count,10,784)
        embds = jnp.matmul(input_array, W1) + b1 # embds.shape=(1,model_count,784,H_in)
        embds = jnp.matmul(W2, embds) + b2
    else:
        W, b = params  # W.shape=(1,model_count,10,784)  b.shape=(1,model_count,10,1)
        if embeddings_type == "not_linear":
            embds = relu_layer([W, b], input_array)
        elif embeddings_type == "linear":
            embds = jnp.matmul(W, input_array) + b
        elif embeddings_type == "pix_to_vec":
            embds = jnp.matmul(input_array, W) + b # W.shape = (1,model_count,1,128) b.shape=(1,model_count,1,128)
        else:
            embds = input_array
    return embds

def rnn_forward_pass(params, linear_recurrent, efficient_rnn_forward_pass, embds, embeddings_type):
    """ Compute the forward pass for each example(sequence) individually """
    M, B, C = params
    seq_length = embds.shape[2] #embs.shape= (num_of_examples,model_count,seq_length,H_in)
    if efficient_rnn_forward_pass and linear_recurrent:
        # the following few lines compute the kernel defined by the RNN
        I_2D = jnp.eye(M.shape[2], M.shape[3])
        num_of_duplicaions = M.shape[0]*M.shape[1]
        I_temp = jnp.tile(I_2D, (num_of_duplicaions, 1, 1))
        I = jnp.reshape(I_temp, M.shape)
        mats = jnp.array([M.squeeze(0)] * (seq_length - 1))  # (embedding_size-1) copies of the matrix M. M.squeeze(0).shape = (model_count,N,N)
        partial_prods = jax.lax.associative_scan(jnp.matmul, mats)  # partial_prods= [M,M*M,M*M*M,M*M*M*M,...]
        partial_prods = jnp.concatenate((I, partial_prods), axis=0)  # [I,M,M*M,M*M*M,M*M*M*M,...]
        res = vmap(lambda A: jnp.matmul(C, jnp.matmul(A, B)))(partial_prods)  # Applying the lambda function on each element in partial_prods [CB,CMB,C(M*M)B,C(M*M*M)B,...]. res.shape=(embedding_size,1,model_count,output_dim,1)
        res_transpose = jnp.transpose(res, axes=(1,2,0,3,4)) # res_transpose.shape=(1,model_count,embeddings_size,output_dim,H_in) = (1,80000,seq_length,2,128)
        expand_embds = jnp.expand_dims(embds, axis=4) # embds.shape=(num_of_samples,model_count,seq_length,H_in,1) = (2,80000,784,128,1). (u_0,u_1,u_2,...u_9)
        reversed_embds = jnp.flip(expand_embds, axis=(2,3,4)) # (u_9,u_8,...,u_0)
        logits = jnp.matmul(res_transpose,reversed_embds).sum(axis=2)  # Calculating RNN outputs  CBu_9 + CMBu_8 ... + C(M^9)Bu_0 - we have a sum of vectors, so the result is a vector.

    else:  # Vanilla RNN
        x = jnp.zeros((1, M.shape[3]))
        x = jnp.expand_dims(x, axis=(0, 1))
        x = jnp.transpose(x, axes=(0, 1, 3, 2))
        for i in range(seq_length):  # Going over all the "words" in the sequence
            input = embds[:, :, i,:]  # input.shape=(num_of_examples,1,1) without embeddings / (num_of_examples,model_count,H_in) with embeddings
            input = jnp.expand_dims(input, axis=3)  # input.shape=(num_of_examples,1,1,1) without embeddings / (num_of_examples,model_count,H_in,1) with embeddings
            if linear_recurrent:
                x = M @ x + B @ input
            else:
                x = jax.nn.sigmoid(M @ x + B @ input)
        logits = C @ x

    logits = logits.real
    return logits  # output shape is (num_of_examples,model_count, output_dim, 1) = (2,80000,2,1)


class RNNModels:
    def __init__(self, N, H_in, H_out, output_dim, r_min, r_max, max_phase, embedding_size, complex, transition_matrix_parametrization, gamma_normalization, official_glorot_init, linear_recurrent, embeddings_type, enable_forward_normalize, num_of_rnn_layers, model_count, scale, efficient_rnn_forward_pass):
        print(f"num_of_rnn_layers={num_of_rnn_layers}")
        if linear_recurrent:
            print("linear")
        else:
            print("not linear")
        if complex:
            print("complex")
        else:
            print("not complex")
        self.N = N
        self.H_in = H_in
        self.H_out = H_out  # intermidate output (used in case of multiple rnn layers)
        self.output_dim = output_dim  # final model output
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase
        self.embedding_size = embedding_size
        self.complex = complex
        self.transition_matrix_parametrization = transition_matrix_parametrization
        self.gamma_normalization = gamma_normalization
        self.official_glorot_init = official_glorot_init
        self.linear_recurrent = linear_recurrent
        self.embeddings_type = embeddings_type
        self.enable_forward_normalize = enable_forward_normalize
        self.num_of_rnn_layers = num_of_rnn_layers
        self.model_count = model_count
        self.scale = scale
        self.efficient_rnn_forward_pass = efficient_rnn_forward_pass
        self.model_key = 1
        self.encoder_layer_params = None
        self.rnn_layers_params = None
        self.encoder_layer_forward_pass = encoder_forward_pass
        self.rnn_layer_forward_pass = rnn_forward_pass
        self.initialize_params()  # initilaizing 'encoder_layer_params' and 'rnn_layers_params' fields

    def generate_layers_keys(self):
        encoder_layer_key = random.PRNGKey(self.model_key)
        rnn_layers_keys = []
        for layer_index in range(self.num_of_rnn_layers):
            rnn_layers_keys.append(random.PRNGKey(self.model_key + 1 + layer_index))  # create for each layer a unique key
        return encoder_layer_key, rnn_layers_keys

    def initialize_encoder_layer_params(self, encoder_layer_key):
        keys = random.split(encoder_layer_key, 4)
        if self.embeddings_type == "pix_to_vec_to_vec":
            params = self.scale * random.normal(keys[0], (1, self.model_count, 1, self.H_in)), self.scale * random.normal(keys[1], (1, self.model_count, 1, self.H_in)), self.scale * random.normal(keys[2], (1, self.model_count, self.embedding_size, 784)), self.scale * random.normal(keys[3], (1, self.model_count, 1, self.H_in))
        elif self.embeddings_type == "pix_to_vec":
            params = self.scale * random.normal(keys[0], (1, self.model_count, 1, self.H_in)), self.scale * random.normal(keys[1], (1, self.model_count, 1, self.H_in)) # H_in=1 embedding_size(seq_length)=128
        else:
            params = self.scale * random.normal(keys[0],(1, self.model_count, self.embedding_size, 784)), self.scale * random.normal(keys[1], (1, self.model_count, self.embedding_size, 1))
        return params

    def initialize_rnn_layer(self, N, H_in, H_out, layer_key, model_count):  # N: state_vector size ,H_in: input size at time k, H_out: output size at time k
        # Glorot initialized Input/Output projection matrices
        keys = random.split(layer_key, 9)
        if self.official_glorot_init:
            B_re = (random.normal(keys[2], (1, model_count, N, H_in))/jnp.sqrt(2*H_in)) * self.scale
            B_im = (random.normal(keys[3], (1, model_count, N, H_in))/jnp.sqrt(2*H_in)) * self.scale
            C_re = (random.normal(keys[4], (1, model_count, H_out, N))/jnp.sqrt(N)) * self.scale
            C_im = (random.normal(keys[5], (1, model_count, H_out, N))/jnp.sqrt(N)) * self.scale
        else: # edo_init
            B_re = random.normal(keys[2], (1, model_count, N, H_in)) * self.scale
            B_im = random.normal(keys[3], (1, model_count, N, H_in)) * self.scale
            C_re = random.normal(keys[4], (1, model_count, H_out, N)) * self.scale
            C_im = random.normal(keys[5], (1, model_count, H_out, N)) * self.scale
        if self.complex:
            B = B_re + 1j * B_im
            C = C_re + 1j * C_im
            if self.transition_matrix_parametrization == "diag_real_im":
                if self.official_glorot_init:
                    A_re_diag = (random.normal(keys[0], (1, model_count, N, 1))/jnp.sqrt(N)) * self.scale # The diagonal of A_re
                    A_im_diag = (random.normal(keys[1], (1, model_count, N, 1))/jnp.sqrt(N)) * self.scale # The diagonal of A_im
                else:
                    A_re_diag = random.normal(keys[0], (1, model_count, N, 1)) * self.scale  # The diagonal of A_re
                    A_im_diag = random.normal(keys[1], (1, model_count, N, 1)) * self.scale  # The diagonal of A_im
                lambda_diag = A_re_diag + 1j * A_im_diag
            else: # "diag_stable_ring_init"
                u1 = random.uniform(keys[7], (1, model_count, N, 1))
                u2 = random.uniform(keys[8], (1, model_count, N, 1))
                nu_log = jnp.log(-0.5 * jnp.log(u1 * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2))
                theta_log = jnp.log(self.max_phase * u2)
                lambda_diag = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
                # gamma Normalization factor
                if self.gamma_normalization:
                    gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(lambda_diag) ** 2))
                    B = B * jnp.exp(gamma_log) # Applying normalization
            M = lambda_diag * jnp.eye(N)

        else:
            if self.official_glorot_init:
                A_re = (random.normal(keys[0], (1, model_count, N, N))/jnp.sqrt(N)) * self.scale
            else:
                A_re = random.normal(keys[0], (1, model_count, N, N)) * self.scale
            M = A_re
            B = B_re
            C = C_re

        return M, B, C

    def initialize_rnn_layers(self, rnn_layers_keys):
        if self.num_of_rnn_layers == 1:
            return [self.initialize_rnn_layer(self.N, self.H_in, self.output_dim, rnn_layers_keys[0], self.model_count)]
        elif self.num_of_rnn_layers == 2:
            return [self.initialize_rnn_layer(self.N, self.H_in, self.H_out, rnn_layers_keys[0], self.model_count)] + [self.initialize_rnn_layer(self.N, self.H_out, self.output_dim, rnn_layers_keys[1], self.model_count)]
        else:
            return [self.initialize_rnn_layer(self.N, self.H_in, self.H_out, rnn_layers_keys[0], self.model_count)] + [self.initialize_rnn_layer(self.N, self.H_out, self.H_out, key, self.model_count) for key in rnn_layers_keys[1:-2]] + [self.initialize_rnn_layer(self.N, self.H_out, self.output_dim, rnn_layers_keys[-1], self.model_count)]

    def initialize_params(self):
        encoder_layer_key, rnn_layers_keys = self.generate_layers_keys()
        self.encoder_layer_params = self.initialize_encoder_layer_params(encoder_layer_key)
        self.rnn_layers_params = self.initialize_rnn_layers(rnn_layers_keys)

    def reinitialize_params(self):
        self.model_key += (self.num_of_rnn_layers + 1)  # update model key to get different random params
        #self.scale = scale
        self.initialize_params()

    def forward(self, input_batch):
        input_batch = jnp.asarray(input_batch.cpu()).reshape(input_batch.shape[0], input_batch.shape[1], -1, 1)  # input_batch.shape = (batch_size,num_of_color_channels,28*28,1)
        embeddings_batch = self.encoder_layer_forward_pass(self.encoder_layer_params, self.embeddings_type, input_batch)  # embeddings_batch.shape=(num_of_examples,1,embeddings_size,1)
        if len(self.rnn_layers_params) > 1:
            output = self.rnn_layer_forward_pass(self.rnn_layers_params[0], self.linear_recurrent, self.efficient_rnn_forward_pass, embeddings_batch , self.embeddings_type)
            for i, params in enumerate(self.rnn_layers_params[1:-2]):
                output = self.rnn_layer_forward_pass(params, self.linear_recurrent, self.efficient_rnn_forward_pass, output, self.embeddings_type)
            output = self.rnn_layer_forward_pass(self.rnn_layers_params[-1], self.linear_recurrent, self.efficient_rnn_forward_pass, output, self.embeddings_type)
        else:
            output = self.rnn_layer_forward_pass(self.rnn_layers_params[0], self.linear_recurrent, self.efficient_rnn_forward_pass, embeddings_batch, self.embeddings_type)

        return output.squeeze(3)  # output.shape= (num_of_examples,model_count,10)

    def weight_normalization(self):
        if self.embeddings_type != "none":
            encoder_layer_normalized_params_list = []
            encoder_params = self.encoder_layer_params
            for param_index, param in enumerate(encoder_params):
                if param_index % 2 == 0:
                    param_norm = jnp.linalg.norm(param, axis=(2, 3), keepdims=True)
                    encoder_layer_normalized_params_list.append(param / param_norm) # W
                else:
                    encoder_layer_normalized_params_list.append(param) # b. bias is not normelize
            self.encoder_layer_params = tuple(encoder_layer_normalized_params_list)
        for layer_num, params in enumerate(self.rnn_layers_params):  # A,B,C = rnn_layers_params
            rnn_layer_normalized_params_list = []
            for param_index, param in enumerate(params):
                param_norm = jnp.linalg.norm(param, axis=(2, 3), keepdims=True)  # Frobenius norm
                rnn_layer_normalized_params_list.append(param / param_norm)

            self.rnn_layers_params[layer_num] = tuple(rnn_layer_normalized_params_list)

    def forward_normalize(self, x):
        if self.enable_forward_normalize:
            self.weight_normalization()
        output = self.forward(x)
        return output

    def get_weights_by_idx(self, idx):
        idx_list = idx.tolist()
        encoder_layer_chosen_params_list = [param[:, idx_list, :, :] for param in self.encoder_layer_params]
        encoder_layer_chosen_params_dict = self.convert_params_list_to_dict(encoder_layer_chosen_params_list, "encoder")
        rnn_layers_chosen_params_list = []
        for layer_params in self.rnn_layers_params:
            layer_chosen_params = [param[:, idx_list, :, :] for param in layer_params]
            rnn_layers_chosen_params_list.append((*layer_chosen_params,))  # converting the params into tuple and appending it to the list
        rnn_layers_chosen_params_dict = self.convert_params_list_to_dict(rnn_layers_chosen_params_list, "rnn")
        return {**encoder_layer_chosen_params_dict,
                **rnn_layers_chosen_params_dict}  # who ever call this function expects to receive a dictionay

    def convert_params_list_to_dict(self, params_list, layer_type):
        params_dict = {}
        if layer_type == "encoder":
            for param_num, param in enumerate(params_list):
                params_dict[f"{layer_type}_param{param_num}"] = param
        else:
            for layer_num, layer_params in enumerate(params_list):
                for param_num, param in enumerate(layer_params):
                    params_dict[f"{layer_type}_layer{layer_num}_param{param_num}"] = param
        return params_dict

    def load_state_dict(self, params_dict):
        if self.embeddings_type == "pix_to_vec_to_vec":
            self.encoder_layer_params = params_dict["encoder_param0"], params_dict["encoder_param1"], params_dict["encoder_param2"], params_dict["encoder_param3"]
            num_of_params_in_single_layer = int((len(params_dict) - 4) / (self.num_of_rnn_layers))
        else:
            self.encoder_layer_params = params_dict["encoder_param0"], params_dict["encoder_param1"]
            num_of_params_in_single_layer = int((len(params_dict) - 2) / (self.num_of_rnn_layers))
        layers_params = []
        for layer_num in range(self.num_of_rnn_layers):
            layer_params_list = []
            for param_num in range(num_of_params_in_single_layer):
                layer_params_list.append(params_dict[f"rnn_layer{layer_num}_param{param_num}"])
            layers_params.append((*layer_params_list,))  # appending current layer params as a tuple

        self.rnn_layers_params = layers_params

    def parameters(self):  # This function returns a list containing all the parameters of the models (including bias)
        if self.embeddings_type == "pix_to_vec_to_vec":
            model_params_list = [self.encoder_layer_params[0], self.encoder_layer_params[1], self.encoder_layer_params[2], self.encoder_layer_params[3]]
        else:
            model_params_list = [self.encoder_layer_params[0], self.encoder_layer_params[1]]
        for layer_params in self.rnn_layers_params:
            for param in layer_params:
                model_params_list.append(param)
        return model_params_list
