# AI vs 弱智吧

- 中文语料库：https://github.com/brightmart/nlp_chinese_corpus?tab=readme-ov-file#1%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91json%E7%89%88wiki2019zh
- embedding 考虑用 https://github.com/shibing624/text2vec

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
