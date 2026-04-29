# ensemble-method-stability-research
This project is also hosted on GitHub: [https://github.com/SkiHatDuckie/ensemble-method-stability-research]()

This project tests the resilience of three different ensemble methods and one
decision tree when noise is increasingly added to a training set. Tests are
done by running `run_tests` with no arguments to get the initial performace on
the clean training set. Then the script is ran again with the `-noise` argument
set to some fraction (ex. `-noise=0.05` -> 5% noise) to test the models
with added label noise, using 3 different methods on sampling. Each combination
of model and sampling method is put through 100 trials on seeds 0->99 to ensure
results aren't based on picking a (un)favorable random seed.

Due to the format of the experiments and how evaluation was done, it did not make
sense for us to merge everything into a single point of execution. Instead, details
for each executable script and its purpose are described in the following section.

`pyproject.toml` is used to handle required dependencies. These dependencies include
`matplotlib`, `pandas`, `scikit-learn`, `seaborn` and `ucimlrepo`.

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