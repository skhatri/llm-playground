curl -H"Authorization: Bearer $HUGGINGFACEHUB_API_TOKEN" -H"Content-Type: application/json" "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1" -d \
'{
  "inputs": "What is the speed of light?"
}'

