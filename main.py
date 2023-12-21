import os
import subprocess
def update_config_file(config_file_path,config_dict):
  with open(config_file_path, 'r') as file:
        lines = file.readlines()
        file.close()
  with open(config_file_path, 'w') as file:
    # for key, value in config_dict.items():
    #   print(f"key={key} value={value}")
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


def fill_config_dict_with_model_params(model_name, config_dict):
    model_name_parts = model_name.split("_")
    config_dict["scale"] = float(model_name_parts[1])
    config_dict["N"] = int(model_name_parts[3])
    config_dict["efficient_rnn_forward_pass"] = True if model_name_parts[6] == "efficient" else False
    config_dict["linear_recurrent"] = True if model_name_parts[9] == "linear" else False
    config_dict["complex"] = True if len(model_name_parts) > 14 else False
    transition_matrix_parametrization = "none"
    r_min = 0
    r_max = 1
    max_phase = 6.28
    gamma_normalization = False
    if config_dict["complex"]:
        transition_matrix_parametrization = "diag_real_im" if model_name_parts[13] == "real" else "diag_stable_ring_init"
        if transition_matrix_parametrization == "diag_stable_ring_init":
            r_min = float(model_name_parts[19])
            r_max = float(model_name_parts[22])
            max_phase = float(model_name_parts[25])
            gamma_normalization = True if model_name_parts[26] == "gamma" else False
    config_dict["transition_matrix_parametrization"] = transition_matrix_parametrization
    config_dict["r_min"] = r_min
    config_dict["r_max"] = r_max
    config_dict["max_phase"] = max_phase
    config_dict["gamma_normalization"] = gamma_normalization

def print_config_details(config_dict):
    H_in = config_dict["H_in"]
    embeddings_type = config_dict["embeddings_type"]
    enable_forward_normalize = config_dict["enable_forward_normalize"]
    scale = config_dict["scale"]
    N = config_dict["N"]
    efficient_rnn_forward_pass = config_dict["efficient_rnn_forward_pass"]
    linear_recurrent = config_dict["linear_recurrent"]
    complex = config_dict["complex"]
    transition_matrix_parametrization = config_dict["transition_matrix_parametrization"]
    r_min = config_dict["r_min"]
    r_max = config_dict["r_max"]
    max_phase = config_dict["max_phase"]
    gamma_normalization = config_dict["gamma_normalization"]
    print(f"DEBUG: H_in={H_in} type(H_in)={type(H_in)}")
    print(f"DEBUG: embeddings_type={embeddings_type} type(embeddings_type)={type(embeddings_type)}")
    print(f"DEBUG: enable_forward_normalize={enable_forward_normalize} type(enable_forward_normalize)={type(enable_forward_normalize)}")
    print(f"DEBUG: scale={scale} type(scale)={type(scale)}")
    print(f"DEBUG: N={N} type(N)={type(N)}")
    print(f"DEBUG: efficient_rnn_forward_pass={efficient_rnn_forward_pass} type(efficient_rnn_forward_pass)={type(efficient_rnn_forward_pass)}")
    print(f"DEBUG: linear_recurrent={linear_recurrent} type(linear_recurrent)={type(linear_recurrent)}")
    print(f"DEBUG: complex={complex} type(complex)={type(complex)}")
    print(f"DEBUG: transition_matrix_parametrization={transition_matrix_parametrization} type(transition_matrix_parametrization)={type(transition_matrix_parametrization)}")
    print(f"DEBUG: r_min={r_min} type(r_min)={type(r_min)}")
    print(f"DEBUG: r_max={r_max} type(r_max)={type(r_max)}")
    print(f"DEBUG: max_phase={max_phase} type(max_phase)={type(max_phase)}")
    print(f"DEBUG: gamma_normalization={gamma_normalization} type(gamma_normalization)={type(gamma_normalization)}")

# tmux methods
def session_exists(session_id):
    result = subprocess.run(["tmux", "has-session", "-t", session_id], stdout=subprocess.PIPE)
    return result.returncode == 0

if __name__ == "__main__":
    config_dict = {}
    config_file_path = "configs/table2/mnist_guess_rnn.yaml"
    tmux_id = os.environ.get("TMUX", " ")[-1]
    if tmux_id == " ":
        tmux_id = "-1"
    config_dict["new_run"] = True
    config_dict["tmux_id"] = tmux_id
    config_dict["embeddings_type"] = "linear"
    config_dict["embedding_size"] = 10
    config_dict["enable_forward_normalize"] = False
    if config_dict["embeddings_type"]=="linear":
      config_dict["H_in"] = 1
    else:
      config_dict["H_in"] = 28

    # models_names = ["scale_0.1_N_32_edo_init_Inefficient_forward_pass_sigmoid_recurrent_rnn_same_seeds",
    #                 "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_rnn_same_seeds",
    #                 "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_real_im_parametrization_rnn",
    #                 "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_6.28_rnn",
    #                 "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_6.28_gamma_normalization_rnn"]
    models_names = ["scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_6.28_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_3.14_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_1.57_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_0.78_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_0.31_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_0.06_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0.8_r_max_1_max_phase_6.28_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0.8_r_max_1_max_phase_3.14_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0.8_r_max_1_max_phase_1.57_gamma_normalization_rnn",
                    "scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0.8_r_max_1_max_phase_0.78_gamma_normalization_rnn"
                    ]
    # models_names = ["scale_0.1_N_32_edo_init_efficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0.8_r_max_1_max_phase_0.31_gamma_normalization_rnn",
    #                 "scale_0.1_N_32_edo_init_efficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0.8_r_max_1_max_phase_0.06_gamma_normalization_rnn"]
    #models_names = ["scale_0.1_N_32_edo_init_Inefficient_forward_pass_linear_recurrent_complex_diag_stable_ring_init_parametrization_r_min_0_r_max_1_max_phase_6.28_gamma_normalization_rnn"]
    command_line = 'python train_distributed_same_seeds.py -C configs/table2/mnist_guess_rnn.yaml'
    # for tmux_session_id, model_name in enumerate(models_names):
    #     if not session_exists(str(tmux_session_id)):
    #         subprocess.run(["tmux", "new-session", "-d", "-s", str(tmux_session_id)])
    model_name = models_names[4]
    tmux_session_id = 4
    print(f"################################## session_id:{tmux_session_id} ##################################")
    fill_config_dict_with_model_params(model_name, config_dict)
    print_config_details(config_dict)
    update_config_file(config_file_path, config_dict)
    subprocess.run(["tmux", "send-keys", "-t", str(tmux_session_id), command_line, "C-m"])


