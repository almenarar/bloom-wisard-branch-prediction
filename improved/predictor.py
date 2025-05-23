import argparse
import csv
import os
import random
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import mmh3
import numpy as np


class BloomFilter:
    def __init__(self, lut_addr_size, num_hashes):
        self.lut = dict()
        self.lut_addr_size = lut_addr_size  # bits
        self.num_hashes = num_hashes
        self.seed = 0

    def add_entry(self, input_piece):
        for i in range(self.num_hashes):
            addr = mmh3.hash(input_piece, self.seed + i) % 2**self.lut_addr_size
            if addr in self.lut:
                self.lut[addr] += 1
            else:
                self.lut[addr] = 1

    def remove_entry(self, input_piece):
        for i in range(self.num_hashes):
            addr = mmh3.hash(input_piece, self.seed + i) % 2**self.lut_addr_size
            if addr in self.lut:
                self.lut[addr] -= 1
                if self.lut[addr] == 0:
                    del self.lut[addr]

    def check_entry(self, input_piece):
        for i in range(self.num_hashes):
            addr = mmh3.hash(input_piece, self.seed + i) % 2**self.lut_addr_size
            if addr not in self.lut:
                return False
        return True

    def binarize(self, b):
        for k in self.lut:
            if self.lut[k] >= b:
                self.lut[k] = 1
            else:
                self.lut[k] = 0
        for k in list(self.lut.keys()):
            if self.lut[k] == 0:
                del self.lut[k]


class Discriminator:
    def __init__(self, num_bloom_filters, lut_addr_size, num_hashes):
        self.bloom_filters = [
            BloomFilter(lut_addr_size, num_hashes) for _ in range(num_bloom_filters)
        ]
        self.num_bloom_filters = num_bloom_filters

    def train(self, input_pieces):
        for i in range(self.num_bloom_filters):
            self.bloom_filters[i].add_entry(input_pieces[i])

    def get_count(self, input_pieces):
        count = 0
        for i in range(self.num_bloom_filters):
            count += int(self.bloom_filters[i].check_entry(input_pieces[i]))
        return count

    def binarize(self, b):
        if b == 0:
            return
        for i in range(self.num_bloom_filters):
            self.bloom_filters[i].binarize(b)

    def forget(self, input_pieces):
        for i in range(self.num_bloom_filters):
            if self.bloom_filters[i].check_entry(input_pieces[i]):
                self.bloom_filters[i].remove_entry(input_pieces[i])


class Model:
    def __init__(self, input_params):
        self.pc_times = input_params[0]
        self.ghr_times = input_params[1]
        self.pc_ghr_times = input_params[2]
        self.lhr1_times = input_params[3]
        self.lhr2_times = input_params[4]
        self.lhr3_times = input_params[5]
        #self.lhr4_times = input_params[6]
        #self.lhr5_times = input_params[7]
        self.ga_times = input_params[6]
        #self.num_bloom_filters = input_params[9]
        #self.num_hashes = input_params[10]
        self.lut_addr_size = input_params[7]
        #self.bleaching_threshold = input_params[12]
        #self.seed = input_params[13]
        self.ghr_size = input_params[8]
        self.ga_branches = input_params[9]
        self.num_bloom_filters = 1
        self.num_hashes = 3
        self.bleaching_threshold = 500
        self.seed = 390

        self.discriminators = [
            Discriminator(self.num_bloom_filters, self.lut_addr_size, self.num_hashes)
            for _ in range(2)
        ]

        # Inicializa o registro de história global
        #self.ghr_size = 24
        self.ghr = np.zeros(self.ghr_size, dtype=np.uint8)

        # Configurações dos registros de história local
        self.lhr_configs = [
            (24, 12),  # (comprimento, bits_pc) para LHR1
            #(16, 10),  # LHR2
            (9, 9),  # LHR3
            #(7, 7),  # LHR4
            (5, 5),  # LHR5
        ]

        # Inicializa os LHRs
        self.lhrs = []
        for length, bits_pc in self.lhr_configs:
            lhr_size = 1 << bits_pc
            self.lhrs.append(np.zeros((lhr_size, length), dtype=np.uint8))

        # Inicializa o endereço global
        self.ga_lower = 8
        #self.ga_branches = 8
        self.ga = np.zeros(self.ga_lower * self.ga_branches, dtype=np.uint8)

        # Calcula o tamanho total da entrada
        self.input_size = (
            self.pc_times * 24
            + self.ghr_times * self.ghr_size
            + self.pc_ghr_times * self.ghr_size
            + sum(self.lhr_configs[i][0] * input_params[i + 2] for i in range(3))
            + self.ga_times * len(self.ga)
        )

    # Extrai características conforme especificado no artigo
    def extract_features(self, pc: int) -> np.ndarray:
        # Extrai bits do PC
        pc_bits = np.array(
            [int(b) for b in format(pc & ((1 << 24) - 1), "024b")], dtype=np.uint8
        )
        pc_bits_repeated = np.tile(pc_bits, self.pc_times)

        # GHR
        ghr_repeated = np.tile(self.ghr, self.ghr_times)

        # PC XOR GHR
        effective_xor_len = min(self.ghr_size, len(pc_bits))
        pc_bits_for_xor = pc_bits[-effective_xor_len:]
        ghr_for_xor = self.ghr[-effective_xor_len:]
        pc_ghr_xor = np.bitwise_xor(pc_bits_for_xor, ghr_for_xor)
        pc_ghr_xor_repeated = np.tile(pc_ghr_xor, self.pc_ghr_times)

        # LHRs
        lhr_features = []
        lhr_times_list = [
            self.lhr1_times,
            self.lhr2_times,
            self.lhr3_times,
            #self.lhr4_times,
            #self.lhr5_times,
        ]
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            if lhr_times_list[i] > 0:
                index = int("".join(map(str, pc_bits[-bits_pc:])), 2)
                lhr = self.lhrs[i][index]
                lhr_repeated = np.tile(lhr, lhr_times_list[i])
                lhr_features.append(lhr_repeated)
        lhr_features_combined = (
            np.concatenate(lhr_features)
            if lhr_features
            else np.array([], dtype=np.uint8)
        )

        # GA
        ga_repeated = (
            np.tile(self.ga, self.ga_times)
            if self.ga_times > 0
            else np.array([], dtype=np.uint8)
        )

        # Combina todas as características
        features = np.concatenate(
            [
                pc_bits_repeated,
                ghr_repeated,
                pc_ghr_xor_repeated,
                lhr_features_combined,
                ga_repeated,
            ]
        )

        return features

    # Atualiza todos os registros de histórico
    def _update_histories(self, pc: int, outcome: int):
        # Atualiza GHR
        self.ghr = np.roll(self.ghr, -1)
        self.ghr[-1] = outcome

        # Atualiza LHRs
        pc_bits = np.array(
            [int(b) for b in format(pc & ((1 << 24) - 1), "024b")], dtype=np.uint8
        )
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int("".join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][index] = np.roll(self.lhrs[i][index], -1)
            self.lhrs[i][index][-1] = outcome

        # Atualiza GA
        new_bits = pc_bits[-self.ga_lower :]
        self.ga = np.roll(self.ga, -self.ga_lower)
        self.ga[-self.ga_lower :] = new_bits

    def get_input_pieces(
        self, features, num_bloom_filters: int, seed: int
    ) -> List[bytes]:
        binary_input = "".join(list(map(str, features.tolist())))
        indices = list(range(len(binary_input)))
        random.seed(seed)
        random.shuffle(indices)
        shuffled_binary = "".join(binary_input[i] for i in indices)
        chunk_size = len(shuffled_binary) // num_bloom_filters
        chunks = [
            shuffled_binary[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_bloom_filters)
        ]
        remainder = len(shuffled_binary) % num_bloom_filters
        for i in range(remainder):
            chunks[i] += shuffled_binary[num_bloom_filters * chunk_size + i]
        return [chunk.encode() for chunk in chunks]

    def predict_and_train(self, pc, outcome):
        features = self.extract_features(pc)
        input_pieces = self.get_input_pieces(
            features, self.num_bloom_filters, self.seed
        )

        count_0 = self.discriminators[0].get_count(input_pieces)
        count_1 = self.discriminators[1].get_count(input_pieces)

        prediction = 0 if count_0 > count_1 else 1

        if prediction != outcome:
            self.discriminators[outcome].train(input_pieces)
            self.discriminators[prediction].forget(input_pieces)

        self._update_histories(pc, outcome)

        return prediction == outcome

    def apply_bleaching(self):
        for disc in self.discriminators:
            disc.binarize(self.bleaching_threshold)


