from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import LLM_MODEL, logger

@lru_cache(maxsize=1)
def get_t5_model():
    logger.info(f"Loading LLM: {LLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

def t5_generate(prompt: str, max_new_tokens: int = 256) -> str:
    tokenizer, model = get_t5_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
