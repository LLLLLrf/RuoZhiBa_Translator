# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")

print(model)
