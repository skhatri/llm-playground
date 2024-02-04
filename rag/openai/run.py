import os
if "OPENAI_API_KEY" not in os.environ:
  print("OPENAI_API_KEY is missing")
  exit(1)

from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('./docs', glob='**/*.md')
docs = loader.load()
print(f"loaded docs {len(docs)}")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
processed_docs = text_splitter.split_documents(docs)

print(f"processed docs vector length {len(processed_docs)}")

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model_name="ada")

query_result = embeddings.embed_query("Hello world")
print(f"embeddings length {len(query_result)}")

import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")
index_name = "doc"

index = Pinecone.from_documents(processed_docs, embeddings, index_name=index_name)
query = "Why do we need Ops Catalog?"
similar_docs = index.similarity_search(query, k=2)
#send all docs as context
similar_docs = processed_docs

from langchain.llms import OpenAI
model_name = "gpt-3.5-turbo-1106"
llm = OpenAI(model_name=model_name)

from langchain.chains.question_answering import load_qa_chain

chain_type = "stuff"
chain = load_qa_chain(llm, chain_type=chain_type)
query = ""
history = {}
while query != "quit" and query != ":q":
  query = input("me: ")
  if query in history:
    answer = history[query]
    print(f"agent: {answer}")
    print("cached info. I saved you few cents or gpu")
  elif query == "":
    print("")
  elif query == "quit" or query == ":q":
    break
  else:
    answer = chain.run(input_documents=similar_docs, question=query)
    print(f"agent: {answer}")
    history[query] = answer



