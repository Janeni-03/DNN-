
# README – RNN, LSTM, GRU and Transformer

## 1. Environment

Experiments were conducted using Google Colab with an NVIDIA Tesla T4 GPU.

| Component | Version / Configuration |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| NumPy | 2.4.6 |
| Pandas | 2.2.3 |
| Scikit-learn | 1.6.1 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| GPU | NVIDIA Tesla T4 |


## 2. How to Run

Open the notebooks using the links below in Google Colab.

### Notebook Links

- [Problem 1 – RNN, LSTM and GRU](Problem1.ipynb)
- [Problem 2 – Transformer from Scratch](Problem2.ipynb)

### Execution Instructions

1. Click the notebook link to open it in GitHub.
2. Select **Open in Colab** (or upload the `.ipynb` file to Google Colab).
3. Select **Runtime → Change runtime type → T4 GPU**.
4. Execute all cells sequentially from top to bottom.

### Problem 1 – RNN, LSTM and GRU

- **Data Pipeline:** Load and preprocess HAR data and generate the Adding Problem dataset.
- **Recurrent Cells:** Implement and verify custom RNN, LSTM, and GRU cells.
- **HAR Classification:** Train and evaluate all three models.
- **Adding Problem:** Evaluate sequence lengths 50, 100, 200, and 400.
- **Gradient and Ablation:** Analyze gradient flow, hidden sizes, and gradient clipping.

### Problem 2 – Transformer from Scratch

- **Attention:** Implement and verify scaled dot-product attention and masking.
- **Transformer Encoder:** Implement multi-head attention, encoder blocks, and positional encoding.
- **HAR Classification:** Train and evaluate the Transformer (executed in Problem 1).
- **Language Model:** Train on Tiny Shakespeare and generate text at different temperatures.
- **Ablation Analysis:** Evaluate attention heads, positional encoding, causal masking, runtime, and memory.

## 3. Execution Time on T4

| Experiment | Execution Time |
|---|---|
| Problem 1 | Approximately 30–60 minutes (estimated) |
| Problem 2 – Tiny Shakespeare training | Approximately 10 minutes (estimated) |
| Problem 2 – Complete runtime | Approximately 15 minutes (estimated) |

## 4. Student's Variant and Implementation

**Student ID:** 2026AJ05924

- Implemented custom `MyRNNCell`, `MyLSTMCell`, and `MyGRUCell` using matrix operations and verified them against PyTorch with differences below 10⁻⁵.
- Implemented Transformer attention, multi-head attention, causal masking, pre-LayerNorm encoder blocks, and positional encoding from scratch.
- Attention verification achieved a maximum difference of approximately 2.38 × 10⁻⁷.

## 5. Main Results

| Experiment | Result |
|---|---:|
| RNN HAR accuracy | 74.54% |
| LSTM HAR accuracy | 88.47% |
| GRU HAR accuracy | 88.15% |
| Transformer training loss | 1.4055 nats |
| Transformer validation loss | 1.5962 nats |
| Tiny Shakespeare training time | 170.53 seconds |

## 6. Known Limitations

- GPU availability and execution time depend on Colab resources.
- Dataset downloads require internet access.
- Long sequences increase computation and memory usage.
- Results may vary across random seeds and hardware.
- Higher generation temperatures reduce text coherence.
- Problem 2 HAR accuracy and some ablation results should be taken from the executed notebooks and CSV files.

## 7. Project Files

| File | Description |
|---|---|
| `Problem1.ipynb` | RNN, LSTM, GRU and Adding Problem experiments |
| `Problem2.ipynb` | Transformer and Tiny Shakespeare experiments |
| `results.csv` | Experimental results |
| `requirements.txt` | Python dependencies |
| `figures/` | Generated plots and visualizations |
| `results/` | Additional experimental outputs |
