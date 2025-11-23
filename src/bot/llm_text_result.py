"""
llm_text_result.py
Responsible only for:
- forming LLM prompt
- calling YandexGPT API
- returning generated text
"""

import json
import os
import requests

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_GPT_ENDPOINT = os.getenv(
    "YANDEX_GPT_ENDPOINT",
    "https://api.ai.yandex.net/large-inference/v1/models/YOUR-MODEL/infer"
)


def get_text(classifications: dict) -> str:
    """
    INPUT: dict of class probabilities
    OUTPUT: disposal instruction text
    """

    prompt = (
        "You are a waste-sorting assistant.\n"
        "Allowed categories: battery, biological, cardboard, clothes, glass, metal, trash, paper, plastic.\n\n"
        f"Detected classifications: {json.dumps(classifications)}\n\n"
        "Generate short instructions:\n"
        "1) A label like 'Likely: plastic (88%)'.\n"
        "2) 2–4 sentences with disposal guidelines.\n"
        "Use simple English.\n"
    )

    # ---------- Fallback if API key not set ---------- #
    if not YANDEX_API_KEY:
        cat, conf = max(classifications.items(), key=lambda kv: kv[1])
        return (
            f"Likely: {cat} ({int(conf * 100)}%).\n"
            f"- Dispose according to local recycling rules.\n"
            f"- (Fallback: Yandex API key not provided)"
        )

    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "input": prompt
        # TODO: Add extra params if required by Yandex API
    }

    try:
        response = requests.post(
            YANDEX_GPT_ENDPOINT,
            headers=headers,
            json=body,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        # TODO: adjust depending on YandexGPT response schema
        text = data.get("output") or data.get("text") or json.dumps(data)

        if isinstance(text, list):
            text = "\n".join(x.get("content", str(x)) for x in text)

        return text

    except Exception as e:
        cat, conf = max(classifications.items(), key=lambda kv: kv[1])
        return (
            f"Likely: {cat} ({int(conf * 100)}%).\n"
            "- Unable to contact YandexGPT.\n"
            f"- Error: {e}\n"
        )
