# REI505M: Brain age prediction from children's EEG

Predicting a child's age from a 15-minute sleep EEG recording (Helsinki Kids 1K dataset, 1032 children aged 0.16 to 16 years). A per-channel 1D CNN reaches a test RMSE of **0.99 years**, against **2.42 years** for a linear regression baseline.

| Model | Val RMSE (years) | Test RMSE (years) |
|---|---|---|
| Mean of log(age) | 4.139 | 4.719 |
| Linear baseline | 1.623 | 2.422 |
| CNN (final model) | 0.820 | 0.991 |


## Notebooks

| Notebook | Contents |
|---|---|
| `eeg_starter/` | Course starter pack: pre-processing, feature examples, data loader (unchanged) |
| `linear-test1.ipynb` | Data exploration (age vs log(age)) and the first linear baseline (one channel, Hjorth features), tests which part of the recording to use (first 300 s) |
| `baseline-bandpower.ipynb` | Linear baseline used: channel C3-P3, Hjorth + band power, val RMSE 1.623 |
| `baseline_random_forest.ipynb` | Random forest baseline, val RMSE 1.676 |
| `CNN_v1.ipynb` | One of the first versions of the CNN |
| `CNN_v2.ipynb` | Another one of the first versions of the CNN |
| `CNN_v3.ipynb` | First two CNN's combined, tests for kernel sizes, full recording, learning-rate schedule, and the final model, val RMSE 0.820 |
| `CNN_v4.ipynb` | Final model with non-overlapping 8 s windows (val RMSE 0.858) |
| `CNN_v5.ipynb` | Final model with 60 s windows and 30 s overlap (val RMSE 1.022) |
| `test_set.ipynb` | Final models on test set + light error analysis |

The trained weights of the final model are in `runs/k15_full_overlap.pt`.

## Reproducibility

All models use seed 1738. The final model and CNN_v4/v5 were trained with deterministic cuDNN, so they reproduce exactly on the same hardware. Earlier CNN_v3 runs were not, so rerunning them gives slightly different numbers. 

Trained on an NVIDIA RTX 2080 Ti
