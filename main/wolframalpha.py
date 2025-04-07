# import requests

# def ask_wolframalpha(query, app_id):
#     url = "http://api.wolframalpha.com/v2/query"
#     params = {
#         "input": query,
#         "appid": app_id,
#         "format": "plaintext",
#     }
#     response = requests.get(url, params=params)
    
#     if response.status_code == 200:
#         # Parse the response
#         data = response.text
#         print(data)
#         return extract_plaintext(data)
#     else:
#         return f"Error: Unable to connect to WolframAlpha. Status Code: {response.status_code}"

# def extract_plaintext(xml_response):
#     # Extract plaintext answers from the XML response
#     import xml.etree.ElementTree as ET
#     root = ET.fromstring(xml_response)
#     results = []
#     for pod in root.findall(".//pod[@title]"):
#         title = pod.attrib.get("title", "")
#         plaintext = pod.findtext(".//plaintext")
#         if plaintext:
#             results.append(f"{title}: {plaintext}")
#     return "\n\n".join(results) if results else "No results found."

# # Replace 'YOUR_APP_ID' with your actual WolframAlpha App ID
# APP_ID = "2KRRUE-G5K6LK5KQ7"
# query = input("What airplanes are flying overhead?")

# response = ask_wolframalpha(query, APP_ID)
# print(response)
import requests
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def ask_wolframalpha(query):
    app_id = os.getenv("WOLFRAM_ALPHA_APP_ID")
    if not query.strip():
        return "Error: Query string is empty. Please provide a valid input."

    url = "http://api.wolframalpha.com/v2/query"
    params = {
        "input": query,
        "appid": app_id,
        "format": "plaintext",
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.text
        return extract_plaintext(data)
    else:
        return f"Error: Unable to connect to WolframAlpha. Status Code: {response.status_code}"

def extract_plaintext(xml_response):
    try:
        root = ET.fromstring(xml_response)
        results = []
        for pod in root.findall(".//pod[@title]"):
            title = pod.attrib.get("title", "")
            plaintext = pod.findtext(".//plaintext")
            if plaintext:
                results.append(f"{title}: {plaintext}")
        return "\n\n".join(results) if results else "No results found."
    except ET.ParseError as e:
        return f"Error: Unable to parse XML response. Details: {e}"

# Replace 'YOUR_APP_ID' with your actual WolframAlpha App ID


# query = "When was Georg Wilhelm Friedrich Hegel born?"  # Ensure input is not empty or only whitespace
#query = "what was Georg Wilhelm Friedrich Hegel's wife name"
# query = "where did  Georg Wilhelm Friedrich Hegel go to school"
# response = ask_wolframalpha(query)
# print("\nWolframAlpha Response:\n")
# print(response)
