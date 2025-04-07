import json
import requests

headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiODgzYWQyYzgtNzEzZi00OGZjLTlkNmEtZmNjMWMyZGZjZjQ4IiwidHlwZSI6ImFwaV90b2tlbiJ9.IN_XYbfmmbXs3-muJVB8wCeSFz7FkMaUw6PxshrJFMo"}

url = "https://api.edenai.run/v2/image/question_answer"
json_payload = {
    "providers": "alephalpha",
    # "file_url": "🔗 URL of your image",
    "question": "When was einstien born ?",
}

response = requests.post(url, json=json_payload, headers=headers)

result = json.loads(response.text)
print(result)
print(result['alephalpha']['answers'])
