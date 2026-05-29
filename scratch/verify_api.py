import urllib.request
import json

r = urllib.request.urlopen('http://localhost:8082/api/cows/0/history')
d = json.loads(r.read())
print(f"History records for Cow #0: {len(d)}")
print(f"First: frame={d[0]['frame_index']}, bbox={d[0]['bbox']}")
print(f"Last:  frame={d[-1]['frame_index']}, bbox={d[-1]['bbox']}")

# Check video endpoint
try:
    req = urllib.request.Request('http://localhost:8082/outputs/test/output_tracked.mp4', method='HEAD')
    vr = urllib.request.urlopen(req)
    print(f"\nVideo endpoint: HTTP {vr.status}")
    print(f"Content-Type: {vr.headers.get('Content-Type')}")
    print(f"Content-Length: {vr.headers.get('Content-Length')} bytes")
except Exception as e:
    print(f"\nVideo endpoint error: {e}")
