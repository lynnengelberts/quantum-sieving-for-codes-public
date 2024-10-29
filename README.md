# Code-based Near Neighbor Search

This project analyzes the asymptotic complexity of **quantum sieving algorithms** combined with **near-neighbor search (NNS)** to solve the decoding problem, which is highly relevant in the context of public-key code-based cryptography. The goal is to compare the performance of various quantum sieving techniques when applied to near-neighbor search. For a detailed explanation and the theoretical background, please refer to the corresponding paper: [https://eprint.iacr.org/2024/1358](https://eprint.iacr.org/2024/1358).

## Project Dependencies

This project is implemented in Python and relies on the following dependencies:

- **math**: for mathematical operations;
- **random**: for generating randomness;
- **scipy**: for numerical optimization;
- **matplotlib**: for data plotting;
- **os** and **csv**: for data storing.

## Algorithm Comparison

To compare the performance of the considered algorithms, you need to set the parameters (as explained in the following subsection) and run the Python script code/main.py. The code provides data on the asymptotic runtime and memory of different algorithms and the corresponding plots that illustrate the algorithms' performance.

### Parameters

The following parameters can be adjusted in the code/main.py script:
- **alg_names**: Names of the algorithms to be compared. Default: ['RPC', 'RPC_Grover', 'RPC_quantum_walk', 'RPC_quantum_walk_sparsification']
- **data_dir**: Directory where the comparison data is stored. Default: ../data/
- **plots_dir**: Directory where the data corresponding plots are saved. Default: ../plots/
- **range_weights**: Number of points at which the complexity is calculated, corresponding to different weights. Default: 100
- **iters**: Number of iterations for which the optimizer runs. Default: 20
- **prec**: Precision of the optimizer. Default: 1e-10

### Running the Comparison

To perform the comparison and generate the results (saved in data_dir) and the corresponding plots (saved in plots_dir), run the main.py script:
```
cd code/
python main.py
```

## Obtaining Numerical Results on Limitations

To obtain numerical data illustrating the limitations of these algorithms, run the code/limitations.py script:

```
cd code/
python limitations.py
```

## Authors
Developed at Centrum Wiskunde & Informatica (CWI) by:
- Lynn Engelberts – Algorithms and Complexity group, QuSoft
- Simona Etinski – Cryptology group
- Johanna Loyer – Cryptology group

## License
This project is licensed under the CC BY-NC License. See the LICENSE file for details.

