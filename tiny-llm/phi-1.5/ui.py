import gradio as gr
import transformers

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


context = """
Introduction: You are David Attenborough. A person knowledeable about Earth and its inhabitants. Whenever you answer questions, show optimism and try to give analogy that exists in Animal Kingdom.
Remember: Remember, you keep your answers no longer than 300 characters.
"""


def generate_with_context(prompt, max_length=1500, bos_token_id=tokenizer.bos_token_id):
    # Combine the context and prompt
    full_prompt = context + "\n\n" + prompt

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    outputs = model.generate(input_ids=inputs["input_ids"], max_length=max_length, bos_token_id=bos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

app = gr.Interface(fn=generate_with_context, inputs="text", outputs="text")

# Start the Gradio app
app.launch()

