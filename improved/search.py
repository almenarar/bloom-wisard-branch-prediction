# Mova a lógica principal da main() para uma função separada
from predictor import Model


def run_predictor_with_params(params: dict, input_file: str) -> float:
    # Mapeie o dicionário 'params' para a lista de parâmetros esperada pelo seu Model
    # Exemplo (ajuste os índices conforme seu Model __init__):
    model_params_list = [
        params['pc_times'],
        params['ghr_times'],
        params['pc_ghr_times'],
        params['lhr1_times'],
        params['lhr2_times'], # ... e assim por diante para todos os lhrN_times
        params['lhr3_times'],
        params['ga_times'],
        params['lut_addr_size'],
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