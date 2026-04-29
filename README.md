# ensemble-method-stability-research

## To Use
From the project directory:
1. Install dependencies with `pip install -e .`
2. Run one of the scripts listed below with `python(3) -m [file] [-args]` (no file extension)

### run_tests
Used for training models and recording performance to a file.
| arguments | type | default | description |
| :--- | :--- | :--- | :--- |
| `-noise` | float (0.0 -- 1.0) | 0.0 | Percentage of noise to inject as a fraction. Noise injection is skipped if set to 0 |
| `-debug` | flag | False | Print results to stdout instead of a file. Purely for debugging |

### evaluate
Generate comparison charts from all available results files in the results folder.
| arguments | type | default | description |
| :--- | :--- | :--- | :--- |
| `-results-files` | str(s) | None | Optional list of result files to use. If omitted, all supported results files in the results directory are loaded |
| `-results-dir` | str | results | Directory containing result files |
| `-output-dir` | str | eval | Directory where charts and summary CSVs are saved |
| `-debug` | flag | False | Print metric summaries for each method and experiment |

### noise_demo
Generates a diagram of the different noise algorithms applied in the experiment.
| arguments | type | default | description |
| :--- | :--- | :--- | :--- |
| `-noise` | float (0.0 -- 1.0) | 0.1 | Percentage of noise to inject as a fraction |

### data_analysis
Various graphs generated during initial data analysis.