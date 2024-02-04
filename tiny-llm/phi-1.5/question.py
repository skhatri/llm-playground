import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name="microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

import sys

context="""Introduction: You are David Attenborough. A person knowledeable about Earth and its inhabitants. Whenever you answer questions, show optimism and try to give analogy that exists in Animal Kingdom.
Remember: Remember, you keep your answers no longer than 300 characters.
"""
raw_input='''
Who discovered Australia?
'''
if len(sys.argv) > 1:
  raw_input=sys.argv[1]
prompt = context + "\n\n" + raw_input
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
import time
start=time.time()

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
end=time.time()
print(f'Time taken: {round(end - start,2)} seconds')

