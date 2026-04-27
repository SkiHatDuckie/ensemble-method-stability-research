# ensemble-method-stability-research

## To Use
From the project directory:
1. Install dependencies with `pip install -e .`
2. Run one of the scripts listed below with `python(3) -m [file] [-args]` (no file extension)

### control_test
| arguments | type | default | description |
| :--- | :--- | :--- | :--- |
| `-noise` | float (0.0 -- 1.0) | 0.0 | Percentage of noise to inject as a fraction. Noise injection is skipped if set to 0 |
| `-debug` | flag | False | Print results to stdout instead of a file. Purely for debugging |

### noise_demo
Generates a diagram of the different noise algorithms applied in the experiment.
| arguments | type | default | description |
| :--- | :--- | :--- | :--- |
| `-noise` | float (0.0 -- 1.0) | 0.1 | Percentage of noise to inject as a fraction |

### data_analysis
Various graphs generated during initial data analysis.