from model import Model

def run_predictor_with_params(params: dict, input_file: str) -> float:
    model_params_list = [
        1,#params['num_pc_filters'],
        1,#params['num_lhr_filters'],
        1,#params['num_ghr_filters'],
        1,#params['num_ga_filters'],
        1,#params['num_xor_filters'], 
        params['pc_lut_addr_size'],
        params['lhr_lut_addr_size'],
        params['ghr_lut_addr_size'],
        params['ga_lut_addr_size'],
        params['xor_lut_addr_size'],
        2000,#params['pc_bleaching_threshold'],
        2000,#params['lhr_bleaching_threshold'],
        2000,#params['ghr_bleaching_threshold'],
        2000,#params['ga_bleaching_threshold'],
        2000,#params['xor_bleaching_threshold'],
        params['pc_tournament_weight'], 
        params['lhr_tournament_weight'],
        params['ga_tournament_weight'],
        params['xor_tournament_weight'],
        params['ghr_tournament_weight'],
        3,#params['pc_num_hashes'],
        3,#params['lhr_num_hashes'],
        3,#params['ghr_num_hashes'],
        3,#params['ga_num_hashes'],
        3,#params['xor_num_hashes'],
        params['ghr_size'],
        params['ga_branches'],
    ]

    predictor = Model(model_params_list) 

    num_branches = 0
    num_predicted = 0
    interval = 2000
    max_lines = 50000
    with open(input_file, "r") as f:
        for line in f:
            if num_branches > max_lines:
                break

            pc, outcome = map(int, line.strip().split())
            num_branches += 1
            if predictor.predict_and_train(pc, outcome):
                num_predicted += 1
            # descomente para ativar o bleaching
            #if num_branches % interval == 0:
            #    predictor.apply_bleaching()

    if num_branches == 0:
        return 0.0 
    return (num_predicted / num_branches) * 100.0 

def fitness_function(individual_params: dict, input_file: str) -> float:
    accuracy = run_predictor_with_params(individual_params, input_file)
    return accuracy