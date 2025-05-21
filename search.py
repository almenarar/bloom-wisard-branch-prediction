# Mova a lógica principal da main() para uma função separada
from model import Model


def run_predictor_with_params(params: dict, input_file: str) -> float:
    # Mapeie o dicionário 'params' para a lista de parâmetros esperada pelo seu Model
    # Exemplo (ajuste os índices conforme seu Model __init__):
    model_params_list = [
        params['num_pc_filters'],
        params['num_lhr_filters'],
        params['num_ghr_filters'],
        params['num_ga_filters'],
        params['num_xor_filters'], # ... e assim por diante para todos os lhrN_times
        params['pc_lut_addr_size'],
        params['lhr_lut_addr_size'],
        params['ghr_lut_addr_size'],
        params['ga_lut_addr_size'],
        params['xor_lut_addr_size'],
        params['pc_bleaching_threshold'],
        params['lhr_bleaching_threshold'],
        params['ghr_bleaching_threshold'],
        params['ga_bleaching_threshold'],
        params['xor_bleaching_threshold'],
        params['pc_tournament_weight'], # Seed fixa para a AG, ou otimize ela tbm
        params['lhr_tournament_weight'],
        params['ga_tournament_weight'],
        params['xor_tournament_weight'],
        params['ghr_tournament_weight'],
        params['pc_num_hashes'],
        params['lhr_num_hashes'],
        params['ghr_num_hashes'],
        params['ga_num_hashes'],
        params['xor_num_hashes'],
        params['ghr_size'],
        params['ga_branches'],
    ]

    # Crie uma instância do Model com esses parâmetros
    predictor = Model(model_params_list) # Ajuste o Model.__init__ para aceitar um dicionário ou lista ordenada

    branches_processed = []
    accuracies = []
    num_branches = 0
    num_predicted = 0
    interval = 10000
    max_lines = 50000
    with open(input_file, "r") as f:
        for line in f:
            if num_branches > max_lines:
                break

            pc, outcome = map(int, line.strip().split())
            num_branches += 1
            if predictor.predict_and_train(pc, outcome):
                num_predicted += 1
            if num_branches % interval == 0:
                predictor.apply_bleaching()

    if num_branches == 0:
        return 0.0 # Evita divisão por zero
    return (num_predicted / num_branches) * 100.0 # Retorna a precisão em %

# A função de aptidão
def fitness_function(individual_params: dict, input_file: str) -> float:
    accuracy = run_predictor_with_params(individual_params, input_file)
    return accuracy