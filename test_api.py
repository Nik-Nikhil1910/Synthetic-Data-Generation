import os
import requests
import json
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def main():
    print("🚀 Starting Gemini REST API test...\n")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ GEMINI_API_KEY is NOT set in this environment")
        sys.exit(1)

    print("✅ GEMINI_API_KEY detected")

    model = "models/gemini-2.0-flash"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Say hello in one short sentence."}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    print("📡 Sending request to Gemini API...\n")

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=30
    )

    print("🔢 HTTP Status Code:", response.status_code)
    print("\n📦 Raw response:")
    print(response.text)

    if response.status_code == 200:
        data = response.json()
        print("\n✅ Parsed model output:")
        print(data["candidates"][0]["content"]["parts"][0]["text"])
    else:
        print("\n❌ Request failed — see error above")


if __name__ == "__main__":
    main()
