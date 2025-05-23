import argparse
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

from model import Model

def main():
    parser = argparse.ArgumentParser(description="BTHOWeN Predictor")
    parser.add_argument("input_file", type=str)
    parser.add_argument("num_pc_filters", type=int, default=2, nargs="?")
    parser.add_argument("num_lhr_filters", type=int, default=2, nargs="?")
    parser.add_argument("num_ghr_filters", type=int, default=2, nargs="?")
    parser.add_argument("num_xor_filters", type=int, default=2, nargs="?")
    parser.add_argument("pc_lut_addr_size", type=int, default=4, nargs="?")
    parser.add_argument("lhr_lut_addr_size", type=int, default=4, nargs="?")
    parser.add_argument("ght_lut_addr_size", type=int, default=4, nargs="?")
    parser.add_argument("xor_lut_addr_size", type=int, default=4, nargs="?")
    parser.add_argument("pc_bleaching_threshold", type=int, default=2, nargs="?")
    parser.add_argument("lhr_bleaching_threshold", type=int, default=2, nargs="?")
    parser.add_argument("ght_bleaching_threshold", type=int, default=2, nargs="?")
    parser.add_argument("xor_bleaching_threshold", type=int, default=2, nargs="?")
    parser.add_argument("pc_tournament_weight", type=int, default=5, nargs="?")
    parser.add_argument("lhr_tournament_weight", type=int, default=2, nargs="?")
    parser.add_argument("ga_tournament_weight", type=int, default=2, nargs="?")
    parser.add_argument("xor_tournament_weight", type=int, default=3, nargs="?")
    parser.add_argument("num_hashes", type=int, default=3, nargs="?")
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
            args.num_pc_filters,
            args.num_lhr_filters,
            args.num_ghr_filters,
            args.num_xor_filters,
            args.pc_lut_addr_size,
            args.lhr_lut_addr_size, 
            args.ght_lut_addr_size,
            args.xor_lut_addr_size,
            args.pc_bleaching_threshold,
            args.lhr_bleaching_threshold,
            args.ght_bleaching_threshold,
            args.xor_bleaching_threshold,
            args.pc_tournament_weight,
            args.lhr_tournament_weight,
            args.ga_tournament_weight,
            args.xor_tournament_weight,
            args.num_hashes,
        ]
        #print(parameters)

    predictor = Model(parameters)
    #print(f"Input size: {predictor.input_size}")

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
                predictor.apply_bleaching()
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
    #print(f"------ Size of each input: {predictor.input_size}")


if __name__ == "__main__":
    main()
