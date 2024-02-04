import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="h2oai/h2o-danube-1.8b-chat",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

context="""Introduction: You are David Attenborough. A person knowledeable about Earth and its inhabitants. Whenever you answer questions, show optimism and try to give analogy that exists in Animal Kingdom.
Remember: Remember, you keep your answers no longer than 300 characters.
"""
import sys
raw_input="Why is drinking water so healthy?"
if len(sys.argv) > 1:
  raw_input=sys.argv[1]

messages = [
    {"role": "system", "content": context},
    {"role": "user", "content": raw_input},
]
prompt = pipe.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
res = pipe(
    prompt,
    max_new_tokens=256,
)
print(res[0]["generated_text"])

