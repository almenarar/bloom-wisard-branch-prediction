import random
import sys
from typing import List, Tuple
import numpy as np

from search import fitness_function

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

HYPERPARAMETER_SPACE = {
    'num_pc_filters': {'type': int, 'min': 2, 'max': 16},
    'num_lhr_filters': {'type': int, 'min': 1, 'max': 16},
    'num_ghr_ga_filters': {'type': int, 'min': 1, 'max': 16},
    'num_xor_filters': {'type': int, 'min': 1, 'max': 16},
    'pc_lut_addr_size': {'type': int, 'min': 1, 'max': 32},
    'lhr_lut_addr_size': {'type': int, 'min': 1, 'max': 32},
    'ght_lut_addr_size': {'type': int, 'min': 1, 'max': 32},
    'xor_lut_addr_size': {'type': int, 'min': 1, 'max': 32},
    'pc_bleaching_threshold': {'type': int, 'min': 0, 'max': 20},
    'lhr_bleaching_threshold': {'type': int, 'min': 0, 'max': 20},
    'ght_bleaching_threshold': {'type': int, 'min': 0, 'max': 20},
    'xor_bleaching_threshold': {'type': int, 'min': 0, 'max': 20},
    'pc_tournament_weight': {'type': float, 'min': 0.0, 'max': 1.0, 'step': 0.05}, # Para pesos, pode usar 'step'
    'lhr_tournament_weight': {'type': float, 'min': 0.0, 'max': 1.0, 'step': 0.05},
    'xor_tournament_weight': {'type': float, 'min': 0.0, 'max': 1.0, 'step': 0.05},
    'ga_tournament_weight': {'type': float, 'min': 0.0, 'max': 1.0, 'step': 0.05},
}
POPULATION_SIZE = 20
NUM_GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 2 # Quantos melhores indivíduos passam direto para a próxima geração

def generate_individual() -> dict:
    individual = {}
    for param, spec in HYPERPARAMETER_SPACE.items():
        if spec['type'] == int:
            individual[param] = random.randint(spec['min'], spec['max'])
        elif spec['type'] == float:
            # Para floats, gere um valor e arredonde para o passo
            val = random.uniform(spec['min'], spec['max'])
            if 'step' in spec:
                individual[param] = round(val / spec['step']) * spec['step']
            else:
                individual[param] = val

    total_weight = individual['pc_tournament_weight'] + individual['lhr_tournament_weight'] + individual['xor_tournament_weight'] + individual['ga_tournament_weight']
    if total_weight > 0:
        individual['pc_tournament_weight'] /= total_weight
        individual['lhr_tournament_weight'] /= total_weight
        individual['xor_tournament_weight'] /= total_weight
        individual['ga_tournament_weight'] /= total_weight
    return individual

def create_initial_population(size: int) -> List[dict]:
    return [generate_individual() for _ in range(size)]

def select_parents(population_with_fitness: List[Tuple[dict, float]]) -> Tuple[dict, dict]:
    # Seleção por roleta ou torneio é comum. Roleta é mais simples para começar.
    # Soma de todas as aptidões para normalização
    total_fitness = sum(f for _, f in population_with_fitness)
    if total_fitness == 0: # Evitar divisão por zero se todas as aptidões forem 0
        return random.choice(population_with_fitness)[0], random.choice(population_with_fitness)[0]

    pick1 = random.uniform(0, total_fitness)
    pick2 = random.uniform(0, total_fitness)
    parent1 = None
    parent2 = None
    current = 0
    for individual, fitness in population_with_fitness:
        current += fitness
        if parent1 is None and current > pick1:
            parent1 = individual
        if parent2 is None and current > pick2:
            parent2 = individual
        if parent1 and parent2:
            break
    return parent1, parent2

def crossover(parent1: dict, parent2: dict) -> Tuple[dict, dict]:
    child1 = parent1.copy()
    child2 = parent2.copy()
    if random.random() < CROSSOVER_RATE:
        # Ponto de crossover aleatório
        keys = list(HYPERPARAMETER_SPACE.keys())
        crossover_point = random.randint(1, len(keys) - 1)
        for i in range(crossover_point):
            key = keys[i]
            child1[key], child2[key] = child2[key], child1[key]

        # Re-normalizar pesos após crossover se eles foram combinados
        if 'pc_tournament_weight' in child1:
            total_weight1 = child1['pc_tournament_weight'] + child1['lhr_tournament_weight'] + child1['ga_tournament_weight'] + child1['xor_tournament_weight']
            total_weight2 = child2['pc_tournament_weight'] + child2['lhr_tournament_weight'] + child2['ga_tournament_weight'] + child2['xor_tournament_weight']
            if total_weight1 > 0:
                child1['pc_tournament_weight'] /= total_weight1
                child1['lhr_tournament_weight'] /= total_weight1
                child1['ga_tournament_weight'] /= total_weight1
                child1['xor_tournament_weight'] /= total_weight1
            if total_weight2 > 0:
                child2['pc_tournament_weight'] /= total_weight2
                child2['lhr_tournament_weight'] /= total_weight2
                child2['ga_tournament_weight'] /= total_weight2
                child2['xor_tournament_weight'] /= total_weight2
    return child1, child2

def mutate(individual: dict) -> dict:
    mutated_individual = individual.copy()
    for param, spec in HYPERPARAMETER_SPACE.items():
        if random.random() < MUTATION_RATE:
            if spec['type'] == int:
                mutated_individual[param] = random.randint(spec['min'], spec['max'])
            elif spec['type'] == float:
                val = random.uniform(spec['min'], spec['max'])
                if 'step' in spec:
                    mutated_individual[param] = round(val / spec['step']) * spec['step']
                else:
                    mutated_individual[param] = val
    # Re-normalizar pesos após mutação
    if 'pc_tournament_weight' in mutated_individual:
        total_weight = mutated_individual['pc_tournament_weight'] + mutated_individual['lhr_tournament_weight'] + mutated_individual['ga_tournament_weight'] + mutated_individual['xor_tournament_weight']
        if total_weight > 0:
            mutated_individual['pc_tournament_weight'] /= total_weight
            mutated_individual['lhr_tournament_weight'] /= total_weight
            mutated_individual['ga_tournament_weight'] /= total_weight
            mutated_individual['xor_tournament_weight'] /= total_weight
    return mutated_individual

def genetic_algorithm(input_file: str):
    num_processes = multiprocessing.cpu_count()
    print(num_processes)

    population = create_initial_population(POPULATION_SIZE)
    best_individual = None
    best_fitness = -1.0

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for generation in range(NUM_GENERATIONS):
            print(f"\n--- Geração {generation + 1}/{NUM_GENERATIONS} ---")

            futures = {
                executor.submit(fitness_function, individual, input_file): individual
                for individual in population
            }

            population_with_fitness = []
            for i, future in enumerate(as_completed(futures)):
                individual = futures[future]
                try:
                    fitness = future.result()
                    population_with_fitness.append((individual, fitness))
                    print(f"  Indivíduo {i+1}: Aptidão = {fitness:.2f}%")

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                except Exception as exc:
                    print(f"  Indivíduo {i+1} gerou uma exceção: {exc}", file=sys.stderr)
                    # O indivíduo que gerou exceção já terá fitness 0.0 da função run_predictor_with_params

            # Ordenar a população por aptidão (do maior para o menor)
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            print(f"Melhor aptidão da geração: {population_with_fitness[0][1]:.2f}%")
            print(f"Melhor aptidão geral até agora: {best_fitness:.2f}%")
            # Para evitar saída muito longa, imprime apenas os melhores parâmetros
            # print(f"Melhores parâmetros até agora: {best_individual}")

            # Criar a próxima geração
            new_population = []
            # Elitismo: Mantém os melhores indivíduos
            for i in range(min(ELITISM_COUNT, POPULATION_SIZE)):
                new_population.append(population_with_fitness[i][0])

            # Preencher o resto da nova população
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = select_parents(population_with_fitness)
                child1, child2 = crossover(parent1, parent2)
                mutated_child1 = mutate(child1)
                mutated_child2 = mutate(child2)
                new_population.append(mutated_child1)
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(mutated_child2)

            population = new_population

    print("\n--- Otimização Concluída ---")
    print(f"Melhores parâmetros encontrados: {best_individual}")
    print(f"Melhor precisão alcançada: {best_fitness:.2f}%")
    return best_individual, best_fitness

# Exemplo de uso:
if __name__ == '__main__':
    # Certifique-se de que este caminho do arquivo de entrada seja válido para o AG
    # Use um subconjunto do seu arquivo de traço para agilizar a otimização!
    # Por exemplo, crie um "trace_sample.txt" com 10.000 a 100.000 linhas
    sample_input_file = "/Users/almenara/Downloads/branch_prediction/Dataset_pc_decimal/I1.txt" 

    # Execute o algoritmo genético
    best_params, final_best_fitness = genetic_algorithm(sample_input_file)

    # Opcional: Execute o preditor com os melhores parâmetros no trace COMPLETO para validação
    #print("\nValidando os melhores parâmetros no trace completo...")
    #full_input_file = "path/to/your/full_trace.txt" # <-- MUDAR AQUI!
    #final_accuracy_full_trace = run_predictor_with_params(best_params, full_input_file)
    #print(f"Precisão final no trace completo com os melhores parâmetros: {final_accuracy_full_trace:.2f}%")