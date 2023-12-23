import os
def update_config_file(config_file_path,config_dict):
  with open(config_file_path, 'r') as file:
        lines = file.readlines()
        file.close()
  with open(config_file_path, 'w') as file:
    #for key, value in config_dict.items():
    # print(f"key={key} value={value}")
    for line in lines:
      line_as_list = line.split(":")
      if not line_as_list: # empty line
        continue
      key = line_as_list[0].strip()
      if key in config_dict.keys():
        value_to_replace = line_as_list[1].strip()
        #print(f"DEBUG: key={key} value_to_replace={value_to_replace} new_val={config_dict[key]}")
        line = line.replace(value_to_replace,str(config_dict[key]))
      file.write(line)
    file.close()


def get_model_name(model_arch,scale,efficient_rnn_forward_pass,linear_recurrent,complex,gamma_normalization,official_glorot_init,transition_matrix_parametrization,N,r_min,r_max,max_phase,embeddings_type,embeddings_size,H_in,dataset_name):
    result = ""
    if model_arch:
        if scale is not None:
            result += f"scale_{scale}_"
        result += f"N_{N}_"
        result += f"H_in_{H_in}_"
        if official_glorot_init:
            result += "paper_"
        else:
            result += "edo_"
        if efficient_rnn_forward_pass:
            result += "efficient_"
        else:
            result += "Inefficient_"
        result += f"embeddings_type_{embeddings_type}_size_{embeddings_size}_"
        if linear_recurrent:
            result += "linear_recurrent_"
        else:
            result += "sigmoid_recurrent_"

        if complex:
            result += f"complex_{transition_matrix_parametrization}_"
            if transition_matrix_parametrization == "diag_stable_ring_init":
                result += f"r_min_{r_min}_r_max_{r_max}_max_phase_{max_phase}_"
        if gamma_normalization:
            result += "gamma_normalization_"
        result += f"{config['dataset.name']}_"
        result += "rnn"
        return result
    else:
        return model_arch


if __name__ == "__main__":
    target_model_count = 1
    linear_recurrent = True
    complex = False
    transition_matrix_parametrization = "diag_real_im"
    gamma_normalization = False
    official_glorot_init = False
    efficient_rnn_forward_pass = False
    scales = [0.01,0.5,0.1,1.0]
    #scales = [0.01]
    embeddings_type = "linear"
    embeddings_size = 784
    loss_thres = "0,10000"
    min_low_loss_bin, max_up_loss_bin = loss_thres.split(",")[0], loss_thres.split(",")[-1]
    if embeddings_type=="linear" or embeddings_type == "none":
      H_in = 1
    elif embeddings_type=="pix_to_vec":
      H_in = 10
    elif embeddings_type=="pix_to_vec_to_vec":
      H_in = 100
    else:
      H_in = 28
    N = 32
    r_min = 0
    r_max = 1
    max_phase = 6.28
    enable_forward_normalize = False
    is_new_run = True
    config_file_path = "configs/table2/mnist_guess_rnn.yaml"

    config_updates_dict = {}
    tmux_id = os.environ.get("TMUX", " ")[-1]
    if tmux_id == " ":
        tmux_id = -1
    config_updates_dict["target_model_count"] = target_model_count
    config_updates_dict["embeddings_type"] = embeddings_type
    config_updates_dict["embedding_size"] = embeddings_size
    config_updates_dict["N"] = N
    config_updates_dict["H_in"] = H_in
    config_updates_dict["r_min"] = r_min
    config_updates_dict["r_max"] = r_max
    config_updates_dict["max_phase"] = max_phase
    config_updates_dict["enable_forward_normalize"] = enable_forward_normalize
    config_updates_dict["linear_recurrent"] = linear_recurrent
    config_updates_dict["complex"] = complex
    config_updates_dict["transition_matrix_parametrization"] = transition_matrix_parametrization
    config_updates_dict["gamma_normalization"] = gamma_normalization
    config_updates_dict["official_glorot_init"] = official_glorot_init
    config_updates_dict["efficient_rnn_forward_pass"] = efficient_rnn_forward_pass
    config_updates_dict["new_run"] = is_new_run
    config_updates_dict["tmux_id"] = tmux_id
    config_updates_dict["loss_thres"] = loss_thres

    for scale in scales:
        model_name = get_model_name("rnn", scale, efficient_rnn_forward_pass, linear_recurrent, complex, gamma_normalization, official_glorot_init, transition_matrix_parametrization,N,r_min,r_max,max_phase,embeddings_type,embeddings_size,H_in)
        print(f"current model name={model_name}")
        cur_file_path = f"output/table2/mnist_guess_rnn/rnn_scale_finding/{model_name}" + f"_loss_bin_range_{min_low_loss_bin},{max_up_loss_bin}.txt"
        config_updates_dict["scale"] = scale
        update_config_file(config_file_path,config_updates_dict)
        if os.path.exists(cur_file_path):
            print(f"experiment:{cur_file_path} already finished!!!")
            continue
        print("#" * 40)
        print("Currently running:")
        print(f"config_path={config_file_path}")
        print(f"new_run={is_new_run} tmux_id={tmux_id} target_model_count={target_model_count} H_in={H_in} embeddings_type={embeddings_type} enable_forward_normalize={enable_forward_normalize}")
        print(f"linear_recurrent={linear_recurrent} complex={complex} transition_matrix_parametrization={transition_matrix_parametrization} gamma_normalization={gamma_normalization} official_glorot_init={official_glorot_init}  scale={scale} ")
        print("#" * 40)
        os.system('python scale_finding.py -C configs/table2/mnist_guess_rnn.yaml')

    scale_lines_count_dict = {}
    for scale in scales:
        cur_lines_counter = 0
        model_name = get_model_name("rnn", scale, efficient_rnn_forward_pass, linear_recurrent, complex, gamma_normalization, official_glorot_init, transition_matrix_parametrization,N,r_min,r_max,max_phase,embeddings_type,embeddings_size,H_in)
        cur_file_path = f"output/table2/mnist_guess_rnn/rnn_scale_finding/{model_name}" + f"_loss_bin_range_{min_low_loss_bin},{max_up_loss_bin}.txt"
        if os.path.exists(cur_file_path):
            with open(cur_file_path,'r') as file:
                for line in file:
                    if line.strip():
                        cur_lines_counter += 1
            scale_lines_count_dict[scale] = cur_lines_counter
        else:
            print(f"{cur_file_path} file is not exists")
    if len(scale_lines_count_dict) < 1:
        print(f"scale_lines_count_dict is empty!!! all scales are bad!!!")
        exit()
    sorted_values = sorted(scale_lines_count_dict.values(),reverse=True)
    best_value = sorted_values[0]
    best_scale = list(scale_lines_count_dict.keys())[list(scale_lines_count_dict.values()).index(best_value)]
    best_str = f"Best scale is: {best_scale} with {best_value} results"
    if len(scale_lines_count_dict) > 1:
        second_best_value = sorted_values[1]
        second_best_scale = list(scale_lines_count_dict.keys())[list(scale_lines_count_dict.values()).index(second_best_value)]
        second_best_str = f"Second best scale is: {second_best_scale} with {second_best_value} results"
    else:
        second_best_str = ""


    Experiment_name = get_model_name("rnn", None, efficient_rnn_forward_pass, linear_recurrent, complex,gamma_normalization, official_glorot_init, transition_matrix_parametrization, N, r_min,r_max, max_phase, embeddings_type,embeddings_size,H_in)
    output_str = Experiment_name + "\n" + best_str + "\n" + second_best_str
    output_path = f'output/table2/mnist_guess_rnn/rnn_scale_finding/best_results_summary/{Experiment_name}_best_results' + f"_loss_bin_range_{min_low_loss_bin},{max_up_loss_bin}.txt"
    print(best_str)
    print(second_best_str)
    with open(output_path, 'w') as file:
        file.write(output_str)










