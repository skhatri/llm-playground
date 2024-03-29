### LLMs

Links to a few papers related to Gen AI

|Papers|
|---|
|[Transformers](transformers.pdf)|
|[Bert](bert.pdf) - Google BERT|
|[Llama2](llama.pdf) - Llama2 |
|[Tiny Llama](tiny-llama.pdf)|
|[Phi 1.5](phi-1.5.pdf) |
|[RAG](rag.pdf)|
|[Active RAG](flare.pdf)|
|[LoRa](lora.pdf)|


### Tiny LLMs
As developers you want to run and see for yourself, Tiny LLM is probably the way to go for enabling LLMs in edge sites and client devices for specialised domains and use cases.

Examples can be run for few tiny LLMs here:

|LLM|Run Instructions|
|---|---|
|Deep Seek - 1.3b parameters|```cd tiny-llm/deepseek && pip install -r requirements.txt && python3 run.py```|
|H2O - Danube 1.8b|```cd tiny-llm/h2o && pip install -r requirements.txt && python3 run.py```|
|Phi-1.5b|```cd tiny-llm/phi-1.5 && pip install -r requirements.txt && python3 question.py && python3 ui.py```|
|Tiny Llama|```cd tiny-llm/tiny-llama && pip install -r requirements.txt && python3 python3 question.py && python3 ui.py```|

### RAG
Retrieval Augmentation Generation using OpenAI endpoint
You need to create OPENAI_API_KEY and PINECONE_API_KEY and make them available as environment variable for this exercise.

```shell
cd rag/openai
pip install -r requirements.txt
python3 run.py
```
The custom data is dropped into rag/openai/docs

![RAG with OpenAI](assets/rag.png)

### Inference HTTP
A basic curl command can be invoked against huggingface model inference endpoints to ask questions quickly.
Some endpoints may need pro subscription

```
./inference/http.sh
```
