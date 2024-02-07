
#https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1
import os

from huggingface_hub.inference_api import InferenceApi
apiToken=os.environ["HUGGINGFACEHUB_API_TOKEN"]
repo="mistralai/Mixtral-8x7B-Instruct-v0.1"
repo="deepset/roberta-base-squad2"
inference = InferenceApi(repo_id=repo, token=apiToken)
inputs = {"question":"Where is Hugging Face headquarters?", "context":"Hugging Face is based in Brooklyn, New York. There is also an office in Paris, France."}
print(inference(inputs))



