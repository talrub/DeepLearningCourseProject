import jax
import numpy
from jax import vmap, random
import jax.numpy as np
from jax.scipy.special import logsumexp
from matplotlib import pyplot as plt
from PIL import Image   
import math
def one_hot(x, k, dtype=np.float32): # x: batch of target_class k: number of classes
    """Create a one-hot encoding of x of size k """
    return np.array(x[:, None] == np.arange(k), dtype) # encoding each target_class in the batch to one-hot vector of size k. 'x[:, None]' reshapes x from (batch,) to (batch,1)

def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return np.maximum(0, x)
    
def relu_layer(params, x):
    """ Simple ReLu layer for single sample """
    return ReLU(np.matmul(params[0], x) + params[1])
    #return ReLU(np.dot(params[0], x) + params[1]()
################################### encoder_forward_pass not using vmap in Vanilla RNN ################################### 
def encoder_forward_pass(params, embeddings_type, input_array):
  input_array_transpose = np.transpose(input_array,axes=(0,1,3,2)) # input_array_transpose = (num_of_examples,1,784,1)
  W, b = params # W.shape=(1,model_count,30,784)  b.shape=(1,model_count,30,1)
  if embeddings_type=="not_linear":
        embds = relu_layer([W, b], input_array_transpose)
  elif embeddings_type=="linear":
        embds = np.matmul(W, input_array_transpose) + b
  else:
        embds = input_array_transpose
  return embds
################################### encoder_forward_pass using vmap in Vanilla RNN################################### 
# def encoder_forward_pass(params, embeddings_type, input_array):
#   input_array= np.expand_dims(input_array, axis=0) # input_array.shape = (1,1,1,784)
#   input_array_transpose = np.transpose(input_array,axes=(0,1,3,2)) # input_array_transpose = (1,1,784,1)
#   W, b = params # W.shape=(1,model_count,30,784)  b.shape=(1,model_count,30,1)
#   if embeddings_type=="not_linear":
#         embds = relu_layer([W, b], input_array_transpose)
#   elif embeddings_type=="linear":
#         embds = np.matmul(W, input_array_transpose) + b
#   else:
#         embds = input_array_transpose
#   embds = np.squeeze(embds,axis=0) # reshape embds to shape (model_count,embedding_size,1)
#   return embds
################################### rnn_forward_pass not using vmap in Vanilla RNN ################################### 
def rnn_forward_pass(params, complex, diagonal ,linear_recurrent, embds):
    """ Compute the forward pass for each example(sequence) individually """
   
    #A_re, A_im, B_re, B_im, C_re, C_im, D = params
    A_re, A_im, B_re, B_im, C_re, C_im = params
    embedding_size = embds.shape[2]
    H_in = B_re.shape[3]
    if complex:
        M = A_re + 1j*A_im
        B = B_re + 1j*B_im
        C = C_re + 1j*C_im
    else:
        M = A_re
        B = B_re
        C = C_re

    if diagonal:
        M = np.diag(M)

    if linear_recurrent:
        # the following few lines compute the kernel defined by the RNN
        I = np.eye(M.shape[0],M.shape[1],M.shape[2],M.shape[3])
        mats = np.array([M]*(embedding_size-1)) # (embedding_size-1) copies of the matrix M
        partial_prods = jax.lax.associative_scan(np.matmul, mats) # partial_prods= [M,M*M,M*M*M,M*M*M*M,...]
        partial_prods = np.concatenate(I, partial_prods) # [I,M,M*M,M*M*M,M*M*M*M,...]

        res = vmap(lambda A: np.matmul(C, np.matmul(A, B)))(partial_prods)  # Applying the lambda function on each element in partial_prods [CAB,C(A*A)B,C(A*A*A)B,...]
        res = res.squeeze() # converting res to 1-D array [CAB,C(A*A)B,C(A*A*A)B,...]
        logits = np.matmul(res.transpose(), embds)  # Calculating RNN outputs  C(A^k)Bx_k + ... + C(A)Bx_0 - we have a sun of vectors, so the result is a vector
    else:  # Vanilla RNN
        x = np.zeros((1,M.shape[3])) #TODO: check with Edo that this x_0 is ok
        x = np.expand_dims(x,axis=(0,1))
        x = np.transpose(x,axes=(0,1,3,2))
        embds = np.reshape(embds,(embds.shape[0],H_in,int(embedding_size/H_in))) # (num_of_examples,1,784,1) or (num_of_examples,1,28,28) according to the choice of H_in
        for i in range(embds.shape[1]): # Going over all the rows of the examples
            input = embds[:, i, :] # input.shape=(num_of_examples,28)
            input = np.expand_dims(input,axis=(1,3))
            B_mul_input = B@input
            M_mul_x = M@x
            x = jax.nn.sigmoid(M@x + B@input)
        logits = C@x
        
    logits = logits.real
    return logits # output shape is (num_of_examples,model_count, H_out, 1) = (2,80,10,1)
################################### rnn_forward_pass using vmap in Vanilla RNN################################### 
# def rnn_forward_pass(params, complex, diagonal ,linear_recurrent, embds):
#     """ Compute the forward pass for each example(sequence) individually """
   
#     #A_re, A_im, B_re, B_im, C_re, C_im, D = params
#     A_re, A_im, B_re, B_im, C_re, C_im = params
#     embedding_size = embds.shape[1]
#     H_in = B_re.shape[3]

#     if complex:
#         M = A_re + 1j*A_im
#         B = B_re + 1j*B_im
#         C = C_re + 1j*C_im
#     else:
#         M = A_re
#         B = B_re
#         C = C_re

#     if diagonal:
#         M = np.diag(M)

#     if linear_recurrent:
#         # the following few lines compute the kernel defined by the RNN
#         I = np.eye(M.shape[0],M.shape[1],M.shape[2],M.shape[3])
#         mats = np.array([M]*(embedding_size-1)) # (embedding_size-1) copies of the matrix M
#         partial_prods = jax.lax.associative_scan(np.matmul, mats) # partial_prods= [M,M*M,M*M*M,M*M*M*M,...]
#         partial_prods = np.concatenate(I, partial_prods) # [I,M,M*M,M*M*M,M*M*M*M,...]

#         res = vmap(lambda A: np.matmul(C, np.matmul(A, B)))(partial_prods)  # Applying the lambda function on each element in partial_prods [CAB,C(A*A)B,C(A*A*A)B,...]
#         res = res.squeeze() # converting res to 1-D array [CAB,C(A*A)B,C(A*A*A)B,...]
#         logits = np.matmul(res.transpose(), embds)  # Calculating RNN outputs  C(A^k)Bx_k + ... + C(A)Bx_0 - we have a sun of vectors, so the result is a vector
#     else:  # Vanilla RNN
#         x = np.zeros((1,M.shape[3])) #TODO: check with Edo that this x_0 is ok
#         x= np.expand_dims(x,axis=(0,1))
#         x = np.transpose(x,axes=(0,1,3,2))
#         embds = np.reshape(embds,(-1,H_in,int(embedding_size/H_in))) # (1,784,1) or (1,28,28) according to the choice of H_in
#         for i in range(embds.shape[1]): # Going over all the rows of the image
#             input = np.transpose(embds[:, i, :]) # input.shape=(28,1)
#             input= np.expand_dims(input,axis=(0,1))
#             B_mul_input = B@input
#             M_mul_x = M@x
#             x = jax.nn.sigmoid(M@x + B@input)
#         logits = C@x
#     print(f"logits.shape={logits.shape}")
#     #print(f"model0_logits={logits.squeeze(0)[0]}")
#     logits = logits.real
#     output = logits.squeeze(0)  # output size should be [model_count, logit_count, 1]
#     return output # output shape is (model_count, H_out, 1) = (80,10,1)

def DEBUG_save_image(array,image_name):
    if torch.is_tensor(array):
        array = array.cpu()
    array_as_np = np.asarray(array).reshape(array.shape[0],array.shape[1],1,-1) # array.shape = (batch_size,num_of_color_channels,1,28*28)
    plt.imshow(array_as_np[0].squeeze(), cmap='gray')
    plt.savefig(image_name)
    #plt.show()
    
    
class RNNModels:
    def __init__(self,N, H_in, H_out, output_dim, embedding_size, complex, diagonal, linear_recurrent, embeddings_type, num_of_rnn_layers, model_count, scale):
        self.N = N
        self.H_in = H_in
        self.H_out = H_out # intermidate output (used in case of multiple rnn layers)
        self.output_dim = output_dim # final model output
        self.embedding_size = embedding_size
        self.complex = complex
        self.diagonal = diagonal
        self.linear_recurrent = linear_recurrent
        self.embeddings_type = embeddings_type
        self.num_of_rnn_layers = num_of_rnn_layers
        self.model_count = model_count
        self.scale = scale
        self.model_key = 1
        self.encoder_layer_params = None
        self.rnn_layers_params = None
        #self.encoder_layer_forward_pass = vmap(encoder_forward_pass, in_axes=(None, None, 0), out_axes=0) # Make a batched version of the `encoder_forward_pas` function. specifing that 'params' and 'embeddings_type' inputs should not be batched and 'in_array' should be batched
        self.encoder_layer_forward_pass = encoder_forward_pass
        #self.rnn_layer_forward_pass = vmap(rnn_forward_pass, in_axes=(None, None, None, None, 0), out_axes=0) # Make a batched version of the `rnn_forward_pass` function. specifing that 'params', 'complex', 'diagonal', 'linear_recurrent', 'last_layer' inputs should not be batched and 'embeds' should be batched
        self.rnn_layer_forward_pass = rnn_forward_pass
        self.initialize_params() # initilaizing 'encoder_layer_params' and 'rnn_layers_params' fields

    def generate_layers_keys(self):
        encoder_layer_key = random.PRNGKey(self.model_key)
        rnn_layers_keys = []
        for layer_index in range(self.num_of_rnn_layers):
            rnn_layers_keys.append(random.PRNGKey(self.model_key + 1 + layer_index)) # create for each layer a unique key
        return encoder_layer_key, rnn_layers_keys

    def initialize_encoder_layer_params(self,encoder_layer_key):
      keys = random.split(encoder_layer_key, 2)
      W, b = self.scale * random.normal(keys[0], (1,self.model_count,self.embedding_size,784)), self.scale * random.normal(keys[1], (1,self.model_count,self.embedding_size,1))
      return W, b

    def initialize_rnn_layer(self,N, H_in, H_out, layer_key, model_count): # N: state_vector size ,H_in: input size at time k, H_out: output size at time k
      # Glorot initialized Input/Output projection matrices
      keys = random.split(layer_key, 7)
      if self.diagonal:
          A_re = random.normal(keys[0], (1,model_count,N,)) * self.scale
          A_im = random.normal(keys[1], (1,model_count,N,)) * self.scale
      else:
          A_re = random.normal(keys[0], (1,model_count,N,N)) * self.scale
          A_im = random.normal(keys[1], (1,model_count,N,N)) * self.scale
     
      B_re = random.normal(keys[2], (1,model_count,N,H_in)) * self.scale
      B_im = random.normal(keys[3], (1,model_count,N,H_in)) * self.scale
      
      C_re = random.normal(keys[4], (1,model_count,H_out,N)) * self.scale
      C_im = random.normal(keys[5], (1,model_count,H_out,N)) * self.scale
      # D = random.normal(keys[6], (1,model_count,H_out,)) * self.scale #TODO:check with Edo if we need to use D
      return A_re, A_im, B_re, B_im, C_re, C_im
      #return A_re, A_im, B_re, B_im, C_re, C_im, D

    def initialize_rnn_layers(self,rnn_layers_keys):
        if self.num_of_rnn_layers == 1:
            return [self.initialize_rnn_layer(self.N, self.H_in, self.output_dim, rnn_layers_keys[0], self.model_count)]
        elif self.num_of_rnn_layers == 2:
            return [self.initialize_rnn_layer(self.N, self.H_in, self.H_out, rnn_layers_keys[0], self.model_count)] + [self.initialize_rnn_layer(self.N, self.H_out, self.output_dim, rnn_layers_keys[1], self.model_count)]
        else:
            return [self.initialize_rnn_layer(self.N, self.H_in, self.H_out, rnn_layers_keys[0], self.model_count)] + [self.initialize_rnn_layer(self.N, self.H_out, self.H_out, key, self.model_count) for key in rnn_layers_keys[1:-2]] + [self.initialize_rnn_layer(self.N, self.H_out, self.output_dim, rnn_layers_keys[-1], self.model_count)]

    def initialize_params(self):
        encoder_layer_key, rnn_layers_keys  = self.generate_layers_keys()
        self.encoder_layer_params = self.initialize_encoder_layer_params(encoder_layer_key)
        self.rnn_layers_params = self.initialize_rnn_layers(rnn_layers_keys)

    def reinitialize_params(self,scale):
        self.model_key += (self.num_of_rnn_layers+1) # update model key to get different random params
        self.scale =  scale
        self.initialize_params()

    def forward(self, input_batch):
        input_batch = np.asarray(input_batch.cpu()).reshape(input_batch.shape[0],input_batch.shape[1],1,-1) # input_batch.shape = (batch_size,num_of_color_channels,1,28*28)
        embeddings_batch =  self.encoder_layer_forward_pass(self.encoder_layer_params,self.embeddings_type,input_batch) # embeddings_batch.shape=(num_of_examples,1,embeddings_size,1)
        #self.DEBUG_compare_models_params(0,1)
        if len(self.rnn_layers_params) > 1:
            output = self.rnn_layer_forward_pass(self.rnn_layers_params[0], self.complex, self.diagonal, self.linear_recurrent, embeddings_batch)
            for i,params in enumerate(self.rnn_layers_params[1:-2]):
                output = self.rnn_layer_forward_pass(params, self.complex, self.diagonal, self.linear_recurrent, output)
            output = self.rnn_layer_forward_pass(self.rnn_layers_params[-1], self.complex, self.diagonal, self.linear_recurrent, output)
        else:
            output = self.rnn_layer_forward_pass(self.rnn_layers_params[0], self.complex, self.diagonal, self.linear_recurrent, embeddings_batch)
        
        return output.squeeze(3) # output.shape= (num_of_examples,model_count,10)
    
    
    def weight_normalization(self):
        if self.embeddings_type != "none":
            W = self.encoder_layer_params[0]
            W_norm = np.linalg.norm(W, axis=(2, 3), keepdims=True)
            self.encoder_layer_params[0] /= W_norm
        for layer_num,params in enumerate(self.rnn_layers_params):  # A_re, A_im, B_re, B_im, C_re, C_im, D = rnn_layers_params
            rnn_layer_normalized_params_list = []
            for param_index, param in enumerate(params):
                param_norm = np.linalg.norm(param, axis=(2, 3), keepdims=True) # Frobenius norm 
                rnn_layer_normalized_params_list.append(param/param_norm)
            
            self.rnn_layers_params[layer_num] = tuple(rnn_layer_normalized_params_list)
                
                
    def forward_normalize(self, x):
        #self.weight_normalization()
        normalized_output = self.forward(x)
        return normalized_output
        

    def get_weights_by_idx(self, idx):
      idx_list = idx.tolist()
      encoder_layer_chosen_params_list = [param[:,idx_list,:,:] for param in self.encoder_layer_params]
      encoder_layer_chosen_params_dict = self.convert_params_list_to_dict(encoder_layer_chosen_params_list,"encoder")
      rnn_layers_chosen_params_list = []
      for layer_params in self.rnn_layers_params:
        layer_chosen_params = [param[:,idx_list,:,:] for param in layer_params]
        rnn_layers_chosen_params_list.append((*layer_chosen_params,)) # converting the params into tuple and appending it to the list
      rnn_layers_chosen_params_dict = self.convert_params_list_to_dict(rnn_layers_chosen_params_list,"rnn")
      return {**encoder_layer_chosen_params_dict,**rnn_layers_chosen_params_dict}  # who ever call this function expects to receive a dictionay 
    
    
    def convert_params_list_to_dict(self,params_list,layer_type):
      params_dict = {}
      if layer_type == "encoder":
        for param_num, param in enumerate(params_list):
          params_dict[f"{layer_type}_param{param_num}"] = param
      else:
        for layer_num, layer_params in enumerate(params_list):
          for param_num, param in enumerate(layer_params):
            params_dict[f"{layer_type}_layer{layer_num}_param{param_num}"] = param
      return params_dict
    

    def load_state_dict(self,params_dict):
      self.encoder_layer_params = params_dict["encoder_param0"], params_dict["encoder_param1"]
      num_of_params_in_single_layer = int((len(params_dict)-2)/(self.num_of_rnn_layers))
      layers_params = []
      for layer_num in range(self.num_of_rnn_layers):
        layer_params_list = []
        for param_num in range(num_of_params_in_single_layer):
          layer_params_list.append(params_dict[f"rnn_layer{layer_num}_param{param_num}"]) 
        layers_params.append((*layer_params_list,)) # appending current layer params as a tuple
        
      self.rnn_layers_params = layers_params
     
      
    def parameters(self): # This function returns a list containing all the parameters of the models (including bias)
        model_params_list = [self.encoder_layer_params[0],self.encoder_layer_params[1]]
        for layer_params in self.rnn_layers_params:
          for param in layer_params:
            model_params_list.append(param)
        return model_params_list
        
        
    def DEBUG_print_model_params(self,model_idx,params_indices):
        print("#"*30)
        print("#"*30)
        if self.embeddings_type != "none":
            print(f"printing encoder layer params:")
            for i,param in enumerate(self.encoder_layer_params):
                if i in params_indices:
                    print(f"model{model_idx}_encoder_layer_param{i}={param}")
                    print("-"*30)
        
        print(f"printing rnn layers params:")
        for layer_num, layer_params in enumerate(self.rnn_layers_params):
            for param_num, param in enumerate(layer_params):
                if param_num in params_indices:
                    print(f"model{model_idx}_layer{layer_num}_param{param_num}={param[:,model_idx,:,:]}")
                    print("-"*30)
        
        print("#"*30)
        print("#"*30)
                    
    
    def DBUG_print_model_params_stats(self,model_idx):
        print("#"*30)
        print("#"*30)
        if self.embeddings_type != "none":
            print(f"printing encoder layer params:")
            for i,param in enumerate(self.encoder_layer_params):
                param_abs = np.abs(param)
                print(f"model{model_idx}_encoder_layer_param{i}: abs_min={np.min(param_abs[:,model_idx,:,:])} abs_max={np.max(param_abs[:,model_idx,:,:])} abs_mean={np.mean(param_abs[:,model_idx,:,:])}")
                print(f"model{model_idx}_encoder_layer_param{i}: min={np.min(param[:,model_idx,:,:])} max={np.max(param[:,model_idx,:,:])} mean={np.mean(param[:,model_idx,:,:])}")
                print("-"*30)
        
        print(f"printing rnn layers params:")
        for layer_num, layer_params in enumerate(self.rnn_layers_params):
            for param_num, param in enumerate(layer_params):
                param_abs = np.abs(param)
                print(f"model{model_idx}_layer{layer_num}_param{param_num}: abs_min={np.min(param_abs[:,model_idx,:,:])} abs_max={np.max(param_abs[:,model_idx,:,:])} abs_mean={np.mean(param_abs[:,model_idx,:,:])}")
                print(f"model{model_idx}_layer{layer_num}_param{param_num}: min={np.min(param[:,model_idx,:,:])} max={np.max(param[:,model_idx,:,:])} mean={np.mean(param[:,model_idx,:,:])} std={np.std(param[:,model_idx,:,:])}")
                eigenvalues = np.linalg.svd(param[:,model_idx,:,:].squeeze(0),compute_uv=False)
                print(f"model{model_idx}_layer{layer_num}_param{param_num}: spectral_norm={np.max(eigenvalues)} frobenius_norm={np.sqrt(np.sum(np.square(param[:,model_idx,:,:])))}")
                print("-"*30)
        
        print("#"*30)
        print("#"*30)
        
    
    def DEBUG_compare_models_params(self,model1_index,model2_index):
        models_are_identical = True
        for layer_num, layer_params in enumerate(self.rnn_layers_params):
            for param_num, param in enumerate(layer_params):
                if np.array_equal(param[:,model1_index,:,:],param[:,model2_index,:,:]):
                    print(f"DEBUG_compare_params : Both models have the same Layer{layer_num}Param{param_num}")
                else:
                    models_are_identical = False
        print(f"DEBUG_compare_models_params: comparing model{model1_index} and model{model2_index} : Are the Models identical? Answer:{models_are_identical}")
    
    def DEBUG_get_number_of_bad_models(self,output): 
        print(f"DEBUG_get_number_of_bad_models: output.shape={output.shape}")
        if output.shape[0] > 1: # output.shape = (num_of_examples,model_count,output_dim)
            comparison_result = np.all(output[0]==output[1], axis=1)
            bad_indices = np.where(comparison_result)[0]
            print(f"$$$$$$$$$$$$$$$$$$$$$ Number of bad models:{len(bad_indices)} $$$$$$$$$$$$$$$$$$$$$")
            print(f"$$$$$$$$$$$$$$$$$$$$$ Priniting bad model parameters statistics (model_idx={bad_indices[0]}): $$$$$$$$$$$$$$$$$$$$$")
            self.DBUG_print_model_params_stats(bad_indices[0])
            for index in range(80000):
                if index not in bad_indices:
                    print(f"$$$$$$$$$$$$$$$$$$$$$ Priniting good model parameters statistics (model_idx={index}): $$$$$$$$$$$$$$$$$$$$$")
                    self.DBUG_print_model_params_stats(index)
                    break
            return len(bad_indices)
        
        
        
            
