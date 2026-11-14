# DNN Programming Assignment-2 – RNN, LSTM, GRU and Transformer

**Student ID:** 2026AJ05924

## 1. Environment
Google Colab, NVIDIA T4 GPU. NumPy: `2.4.6`, Pandas: `2.2.3`, Scikit-learn: `1.6.1`, Matplotlib: `3.10.0`, Seaborn: `0.13.2`, PyTorch: `2.14.1`. small GPU run-to-run differences are expected.

## 2. How to run
1. **Upload the UCI HAR zip first.** In Colab, open the Files panel (left sidebar), click Upload, and select `har.zip` (the file downloaded from the UCI "Human Activity Recognition Using Smartphones" page). Put it in the same working directory as the notebook, with this exact name. It is not included in my submission because the brief says not to include datasets.
2. Open `problem1/problem1.ipynb` (or `problem2/problem2.ipynb`) in Google Colab.
3. Runtime → Change runtime type → T4 GPU.
4. Runtime → Restart and run all.

The notebook extracts `har.zip` automatically, including the inner `UCI HAR Dataset.zip` if present. Tiny Shakespeare is downloaded automatically (internet needed); the adding-problem data is generated in code. Problem 2 Task A (HAR encoder classifier) is in `problem1/problem1.ipynb` and reuses the Problem 1 data pipeline. Each notebook is submitted with its outputs saved.

## 3. Total run time on a T4 (Restart and Run All)
| Notebook | Time |
|---|---|
| problem1.ipynb | [measured, 60 min] |
| problem2.ipynb | [measured, 28 min] (Tiny Shakespeare training alone: 170.5 s) |

## 4. My variant
- **Hidden size (Problem 1):** 64
- **Transformer width / seeds:** 128 / (0, 1, 2)
- **Problem 1 Task 5(c) ablation:** Gradient clipping (norm 1.0 vs. no clipping) for the vanilla RNN at T = 200.
- **Epoch budget:**
  - Task 5(a): 30 epochs for RNN, LSTM, and GRU.
  - Task 5(b): 20 epochs for LSTM and GRU hidden-size sweeps.
  - Task 5(c): 30 epochs for both clipping conditions.

## 5. Assumptions
- The UCI HAR zip must be uploaded manually to the Colab working directory as `har.zip` before running (see section 2). Colab deletes it when the session ends, so re-upload after a disconnect.
- CUDA was used for GPU acceleration where available.
- Random seeds (0, 1, and 2) were used to evaluate model performance across multiple runs.

### Known Issues
- Model performance may vary slightly depending on random initialization and hardware.
- Transformer models have significantly more trainable parameters than recurrent models and require more computation per epoch.
- Transformer performance can be sensitive to positional encoding, attention configuration, and hyperparameter choices.

## 6. Files
| File | Description |
|---|---|
| `report.pdf` | Report (include problem 1 - 5(d) analysis) |
| `results.csv` | One row per run, in the required columns |
| `requirements.txt` | Exact package versions |
| `problem1/problem1.ipynb`, `problem1/figures/` | RNN/LSTM/GRU notebook and plots |
| `problem2/problem2.ipynb`, `problem2/figures/` | Transformer notebook and plots |
| `problem2/generated_samples.txt` | Three 500-character samples (T = 0.2, 0.8, 1.5) |
