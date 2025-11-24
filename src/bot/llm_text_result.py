# llm_text_result.py
import time
import jwt
import requests
import threading
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID")
SERVICE_ACCOUNT_ID = os.getenv("SERVICE_ACCOUNT_ID")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
KEY_ID = os.getenv("KEY_ID")


class IAMTokenManager:
    """
    Efficient token cache + auto refresh + thread safe
    """
    _token = None
    _expires_at = 0
    _lock = threading.Lock()

    @classmethod
    def get_token(cls) -> str:
        with cls._lock:
            now = time.time()
            # if still valid for ≥ 60 sec — use
            if cls._token and now < cls._expires_at - 60:
                return cls._token

            # refresh
            cls._token, cls._expires_at = cls._generate_token()
            return cls._token

    @classmethod
    def _generate_token(cls):
        now = int(time.time())
        payload = {
            "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            "iss": SERVICE_ACCOUNT_ID,
            "iat": now,
            "exp": now + 3600  # lifetime
        }

        encoded_jwt = jwt.encode(
            payload,
            PRIVATE_KEY,
            algorithm="PS256",
            headers={"kid": KEY_ID},
        )

        resp = requests.post(
            "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            json={"jwt": encoded_jwt},
            timeout=10
        )

        if resp.status_code != 200:
            raise RuntimeError(f"IAM error {resp.status_code}: {resp.text}")

        iam_token = resp.json()["iamToken"]
        expires_at = now + 360
        return iam_token, expires_at



SYSTEM_PROMPT = """
You are a waste-disposal assistant.  
Your job is to give short, practical, human explanations on how to correctly dispose of items detected by a model.  
You receive:  
- one or more waste item names
Your output must be concise, specific, and non-repetitive.

Rules for your response:
1. Use natural conversational human language.
2. Keep it short but meaningful — 4–7 sentences total for all items together.
3. Provide a clear, direct instruction for each item: where it must be thrown away, how to prepare it (clean, separate parts, empty contents, wrap, etc.).
5. If you are not sure about the specific rules of disposal in Russia, DO NOT mention “follow local laws” or other meaningless filler. Instead give the most widely accepted global practice.
6. No lists, no bullet points, no section numbers, no markdown.
7. Avoid repeating identical phrases such as “this item can be recycled” for each item. Merge information naturally.
8. The final answer must be directly useful for a real person standing in front of trash bins.

Your goal is to produce the most practically helpful and specific instruction possible, even when regional rules are unknown.
"""


def call_yandex_gpt(user_text: str, model_name: str) -> str:
    iam_token = IAMTokenManager.get_token()

    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json",
    }

    data = {
        "modelUri": f"gpt://{FOLDER_ID}/{model_name}",
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 600
        },
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": user_text}
        ]
    }

    resp = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        headers=headers,
        json=data,
        timeout=30
    )

    if resp.status_code != 200:
        raise RuntimeError(f"GPT API error {resp.status_code}: {resp.text}")

    raw = resp.json()["result"]["alternatives"][0]["message"]["text"]
    return raw.replace("`", "")


# ==============================
# PUBLIC FUNCTION USED BY BOT
# ==============================

def get_text(classifications: List[Dict]) -> str:
    """
    classifications = [
        {"label": "plastic", "confidence": 0.87},
        {"label": "metal", "confidence": 0.65}
    ]
    """

    text_for_gpt = "Detected items:\n"
    for c in classifications:
        text_for_gpt += f"- {c['label']} (confidence {c['confidence']:.2f})\n"

    print(text_for_gpt)

    return call_yandex_gpt(text_for_gpt, model_name="yandexgpt")
