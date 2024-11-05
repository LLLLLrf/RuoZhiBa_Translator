# AI vs RuoZhiBa

## Transformer

Suppose is done:
- [x] CUDA Environment
- [x] PyTorch
- [x] python
- [x] pip

Install dependencies:

```bash
pip install -r requirements.txt
```

Change hyperparameters in `transformer.py`:

```python
...
# Parameters
model_name = "google/mt5-base"
fold_nums = 10
num_epochs = 100
max_length = 128
batch_size = 8
lr = 1e-4
...
```

Train the model:

```bash
python transformer.py
```

It will save all file in `./runs/` directory, pay attention to the loggings.


## LSTM

Train the model:

```bash
python lstm.py
```

## RNN

Train the model:

```bash
python rnn.py
```