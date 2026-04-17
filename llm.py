import re
import requests

WEBSITE_ONLY_FALLBACK = "This information is not in this website."


def clean_response(text: str) -> str:
    if not text:
        return WEBSITE_ONLY_FALLBACK

    # Remove leaked prompt/instruction text
    stop_patterns = [
        r"User:.*",
        r"Assistant:.*",
        r"STRICT RULES:.*",
        r"Use ONLY the WEBSITE CONTEXT.*",
        r"Do NOT use your own knowledge.*",
        r"Do NOT guess.*",
        r"Do NOT answer from general knowledge.*",
        r"Do NOT add examples.*",
        r"Do NOT continue.*",
        r"WEBSITE CONTEXT:.*",
        r"Question:.*",
        r"Answer:.*"
    ]

    cleaned = text.strip()

    for pattern in stop_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    cleaned = cleaned.strip()

    # Remove repeated spaces/newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # If model starts quoting weirdly
    cleaned = cleaned.strip(' "\'')

    # If response got cut mid-way, trim to last full sentence
    sentence_end = max(
        cleaned.rfind("."),
        cleaned.rfind("!"),
        cleaned.rfind("?")
    )

    if sentence_end > 50:
        cleaned = cleaned[:sentence_end + 1]

    if not cleaned:
        return WEBSITE_ONLY_FALLBACK

    return cleaned


def generate_answer(context: str, question: str, model_name: str = "phi") -> str:
    prompt = f"""
You are a STRICT website-based question answering assistant.

You must answer ONLY using the WEBSITE CONTEXT below.

STRICT RULES:
- Use ONLY the WEBSITE CONTEXT
- Do NOT use your own knowledge
- Do NOT guess
- Do NOT answer from general knowledge
- Do NOT add examples unless they are clearly present in the context
- Do NOT continue the conversation
- Do NOT write User:
- Do NOT write Assistant:
- If the answer is not clearly in the context, respond exactly with:
This information is not in this website.

WEBSITE CONTEXT:
{context}

Question:
{question}

Answer:
""".strip()

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 200,
                    "stop": [
                        "User:",
                        "Assistant:",
                        "Question:",
                        "STRICT RULES:",
                        "WEBSITE CONTEXT:"
                    ]
                }
            },
            timeout=120
        )

        data = response.json()

        raw_answer = data.get("response", "").strip()

        if not raw_answer:
            return WEBSITE_ONLY_FALLBACK

        cleaned_answer = clean_response(raw_answer)

        if not cleaned_answer:
            return WEBSITE_ONLY_FALLBACK

        return cleaned_answer

    except Exception:
        return WEBSITE_ONLY_FALLBACK
