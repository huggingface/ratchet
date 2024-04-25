import torch
from transformers import Phi3ForCausalLM, AutoTokenizer
import collections

def ground():
    model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    inputs = tokenizer("def print_prime(n):", return_tensors="pt", return_attention_mask=False)
    logits = model(**inputs).logits
    print("LOGITS: ", logits)

ground()
