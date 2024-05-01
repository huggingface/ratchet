import torch
from transformers import Phi3ForCausalLM, AutoTokenizer

def ground():
    model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    print("Model: ", model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    inputs = tokenizer("""<|user|>
How to explain Internet for a medieval knight?<|end|>
<|assistant|>""", return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=500, return_dict_in_generate=True, output_logits=True)
    tokens = outputs[0]
    print("Tokens: ", tokens)
    print("Text: ", tokenizer.decode(tokens[0], skip_special_tokens=True))

def hooked():
    model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    model.eval()

    first_layer_output = None
    def hook(module, input, output):
        nonlocal first_layer_output
        first_layer_output = output
    
    # Register the forward hook on the first decoder layer
    model.model.layers[0].register_forward_hook(hook)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    inputs = tokenizer('{}', return_tensors="pt", return_attention_mask=False)
    print("PROMPT TOKENS:", inputs["input_ids"])
    logits = model(**inputs).logits
    print("FIRST LAYER OUTPUT: ", first_layer_output)
    return [first_layer_output[0].detach().numpy()]


ground()
