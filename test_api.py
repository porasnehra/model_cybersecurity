import requests
import json
import random

# We are going to send some random dummy data to the API to test it
url = "http://localhost:8000/predict"

# Constructing a dummy feature dictionary
# The model expects numeric features, missing ones are handled by the imputer
dummy_features = {}
for i in range(1, 100): # Just sending the first 100 features as an example
    dummy_features[f"F{i}"] = random.uniform(0, 100)

payload = {
    "features": dummy_features
}

headers = {
    'Content-Type': 'application/json'
}

print(f"Sending POST request to {url}...")
try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print("\nStatus Code:", response.status_code)
    print("Response Body:")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.ConnectionError:
    print("\nError: Could not connect to the API.")
    print("Did you remember to start the server using 'uvicorn app:app --reload' ?")
