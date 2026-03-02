import requests
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing OpenAI API connectivity...")
try:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "say hi"}],
            "temperature": 0,
        },
        timeout=15,
    )
    print(f"Status: {r.status_code}")
    print(r.json())
except requests.exceptions.Timeout:
    print("TIMED OUT — API not reachable within 15 seconds")
except Exception as e:
    print(f"ERROR: {e}")