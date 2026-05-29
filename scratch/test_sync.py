import json
import urllib.request

def main():
    url = "http://localhost:8082/api/upload"
    payload = [{
        "frame_index": 0,
        "cattle_id": 99,
        "bbox": [100, 200, 300, 400],
        "status": "Estrus"
    }]
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        print("Sending request...")
        with urllib.request.urlopen(req, timeout=5.0) as response:
            status = response.status
            body = response.read().decode('utf-8')
            print(f"Response status: {status}")
            print(f"Response body: {body}")
    except Exception as e:
        print(f"Exception during request: {e}")

if __name__ == "__main__":
    main()
