import requests
import json

MY_API_KEY = "fe3e1a8e23b80ee1ab9a75610a2a0b108afdd61ab2b9c3c59c87e50d"
API_URL = "https://my-api-project-kvz4.onrender.com/predict" # Change to your public URL

features_data = {
  "features": [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601, 0.5
  ]
}

headers = {
    "Content-Type": "application/json",
    "X-API-Key": MY_API_KEY
}

response = requests.post(API_URL, headers=headers, data=json.dumps(features_data))

if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Error:", response.status_code, response.json())