import requests

try:
    response = requests.get("http://localhost:8000/api/predict?stock_symbol=TCS")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
