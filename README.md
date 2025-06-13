# Bloom WiSARD Branch Prediction with Tournament

This repository implements a branch prediction using a tournament-based architecture and Bloom WiSARD, with support for genetic algorithm-based hyperparameter optimization. The main logic is contained in the [`tournament`](tournament/) directory.

## Directory Structure

- [`tournament/`](tournament/): Main implementation of the tournament predictor and supporting modules.
  - [`main.py`](tournament/main.py): Entry point for running the predictor on a dataset.
  - [`model.py`](tournament/model.py): Defines the `Model` class, which implements the tournament predictor logic, feature extraction, and prediction/training routines.
  - [`bloom_filter.py`](tournament/bloom_filter.py): Implements a Bloom filter used by discriminators for fast membership checking.
  - [`discriminator.py`](tournament/discriminator.py): Implements discriminators that use Bloom filters for prediction.
  - [`ag.py`](tournament/ag.py): Implements a genetic algorithm for hyperparameter optimization.
  - [`search.py`](tournament/search.py): Contains the fitness function and logic for evaluating predictor configurations.
- [`Dataset_pc_decimal/`](Dataset_pc_decimal/): Datasets of third branch prediction championship 

## How It Works

- The predictor uses multiple discriminators (PC, LHR, GHR, GA, XOR) to make predictions, each with its own Bloom filter.
- Features are extracted from program counters and history registers, split into pieces, and hashed into the Bloom filters.
- The final prediction is made using a weighted tournament of the discriminators' votes.
- The genetic algorithm (`ag.py`) can be used to optimize hyperparameters such as LUT address sizes, tournament weights, and history lengths.

## Usage

### Running the Predictor

To run the predictor on a dataset:

```sh
python main.py <input_file> <pc_lut_addr_size> <lhr_lut_addr_size> <ght_lut_addr_size> <ga_lut_addr_size> <xor_lut_addr_size> <pc_tournament_weight> <lhr_tournament_weight> <ghr_tournament_weight> <ga_tournament_weight> <xor_tournament_weight> <ghr_size> <ga_branches>
```

<input_file>: Path to the dataset file (e.g., Dataset_pc_decimal/I1.txt).
The remaining arguments are hyperparameters for the model.


## Hyperparameter Optimization

To optimize hyperparameters using the genetic algorithm, use ag.py:

```sh
python ag.py <input_file>
```

This will search for the best set of parameters for the given dataset.

### Output

Accuracy results and plots are saved in the `Results_accuracy/`directory.
Final accuracy values are also appended to `true_bthowen_accuracy/`.


## Requirements

- Python 3
- numpy
- matplotlib
- mmh3

Install dependencies with:

```sh
pip install numpy matplotlib mmh3
```

## References

- Aleksander, Igor, et al. "A brief introduction to weightless neural systems." ESANN. 2009.
- [`SANTIAGO, Leandro et al. Memory Efficient Weightless Neural Network using Bloom Filter. 2018.`](https://www.esann.org/sites/default/files/proceedings/legacy/es2019-83.pdf)
- [`Villon, Luis AQ, et al. "A conditional branch predictor based on weightless neural networks." Neurocomputing 555 (2023): 126637.`](https://www.sciencedirect.com/science/article/pii/S0925231223007609)