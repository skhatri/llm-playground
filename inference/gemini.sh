curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=$GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{
          "text": "List the name of Planets in our solar system and their distance from the sun."}]}]}' 2> /dev/null

