import torch
import time
import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer

model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

context="""Introduction: You are David Attenborough. A person knowledeable about Earth and its inhabitants. Whenever you answer questions, show optimism and try to give analogy that exists in Animal Kingdom.
Remember: Remember, you keep your answers no longer than 300 characters.
"""

def generate_with_context(prompt, max_length=1500, bos_token_id=tokenizer.bos_token_id):
    messages = [
      {
          "role": "system",
          "content": context
      },
      {"role": "user", "content": prompt},
    ]

    start = time.time()
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    end = time.time()
    print(f'Time: {round(end - start, 2)} seconds')
    return outputs[0]["generated_text"]

app = gr.Interface(fn=generate_with_context, inputs="text", outputs="text")

app.launch()





