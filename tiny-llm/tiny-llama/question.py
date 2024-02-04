import torch
import time
import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer

model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

raw_input="Who discovered Australia?"

import sys
if len(sys.argv) > 1: 
  raw_input=sys.argv[1]

context="""Introduction: You are David Attenborough. A person knowledeable about Earth and its inhabitants. Whenever you answer questions, show optimism and try to give analogy that exists in Animal Kingdom.
Remember: Remember, you keep your answers no longer than 300 characters.
"""

messages = [
      {
          "role": "system",
          "content": context, 
      },
      {"role": "user", "content": raw_input},
]

start = time.time()
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
end = time.time()
print(f'Time: {round(end - start, 2)} seconds')
print(outputs[0]["generated_text"])







