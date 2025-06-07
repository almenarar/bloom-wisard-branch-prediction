import argparse
import csv
import os
import random
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import mmh3
import numpy as np

from predictor import Model

def main():
    parser = argparse.ArgumentParser(description="BTHOWeN Predictor")
    parser.add_argument("input_file", type=str)
    parser.add_argument("pc_times", type=int, default=1, nargs="?")
    parser.add_argument("ghr_times", type=int, default=1, nargs="?")
    parser.add_argument("pc_ghr_times", type=int, default=1, nargs="?")
    parser.add_argument("lhr1_times", type=int, default=1, nargs="?")
    parser.add_argument("lhr2_times", type=int, default=1, nargs="?")
    parser.add_argument("lhr3_times", type=int, default=1, nargs="?")
    parser.add_argument("ga_times", type=int, default=1, nargs="?")
    parser.add_argument("lut_addr_size", type=int, default=14, nargs="?")
    parser.add_argument("ghr_size", type=int, default=0, nargs="?")
    parser.add_argument("ga_branches", type=int, default=0, nargs="?")
    parser.add_argument("--params_file", type=str)

    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Error: The file '{args.input_file}' does not exist.")

    if args.params_file:
        if not os.path.isfile(args.params_file):
            raise FileNotFoundError(
                f"Error: The parameters file '{args.params_file}' does not exist."
            )
        with open(args.params_file, "r") as f:
            parameters = list(map(int, f.read().strip().split()))
        if len(parameters) != 12:
            raise ValueError(
                "Error: The parameters file must contain exactly 12 integers."
            )
    else:
        parameters = [
            args.pc_times,
            args.ghr_times,
            args.pc_ghr_times,
            args.lhr1_times,
            args.lhr2_times,
            args.lhr3_times,
            #args.lhr4_times,
            #args.lhr5_times,
            args.ga_times,
            args.lut_addr_size,
            args.ghr_size,
            args.ga_branches,
        ]

    predictor = Model(parameters)
    print(f"Input size: {predictor.input_size}")

    num_branches = 0
    num_predicted = 0
    interval = 10000

    branches_processed = []
    accuracies = []
    with open(input_file, "r") as f:
        for line in f:
            pc, outcome = map(int, line.strip().split())
            num_branches += 1
            if predictor.predict_and_train(pc, outcome):
                num_predicted += 1
            if num_branches % interval == 0:
                accuracy = (num_predicted / num_branches) * 100
                branches_processed.append(num_branches)
                accuracies.append(accuracy)
                #predictor.apply_bleaching()
                print(f"Branch number: {num_branches}")
                print(f"----- Partial Accuracy: {accuracy:.4f}\n")

    input_file_base = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = f"Results_accuracy/{input_file_base}"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    plt.figure(figsize=(10, 6))
    plt.plot(branches_processed, accuracies, marker="o")
    plt.title("Accuracy Over Time")
    plt.xlabel("Number of Branches Processed")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.savefig(f"{output_dir}/{timestamp}-BTHOWeN-accuracy.png")
    # plt.show()

    final_accuracy = (num_predicted / num_branches) * 100

    os.makedirs("true_bthowen_accuracy", exist_ok=True)

    with open(f"{output_dir}/{timestamp}-BTHOWeN-accuracy.csv", "w", newline="") as e:
        writer = csv.writer(e)
        writer.writerow(
            ["Number of Branches Processed", "Accuracy (%)"]
        )
        writer.writerows(zip(branches_processed, accuracies))  # Dados do gráfico

    with open(f"true_bthowen_accuracy/{input_file_base}-accuracy.csv", "a") as f:
        f.write(f"{final_accuracy:.4f},BTHOWeN,{','.join(list(map(str,parameters)))}\n")

    print("\n----- Results ------")
    print(f"Predicted branches: {num_predicted}")
    print(f"Not predicted branches: {num_branches - num_predicted}")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"\n------ Size of ntuple (address_size): {parameters[0]}")
    print(f"------ Size of each input: {predictor.input_size}")


if __name__ == "__main__":
    main()
