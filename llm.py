import requests

WEBSITE_ONLY_FALLBACK = "This information is not in the website."


def clean_response(text: str) -> str:
    if not text:
        return WEBSITE_ONLY_FALLBACK

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    cleaned = "\n".join(unique_lines).strip()
    return cleaned if cleaned else WEBSITE_ONLY_FALLBACK


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
- Do NOT continue the prompt
- If the answer is not clearly present in the WEBSITE CONTEXT, reply with exactly:
This information is not in the website.
- For summary questions, summarize only the main topics clearly present in the context
- Keep the answer concise and grounded

WEBSITE CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "num_predict": 140,
                    "stop": ["USER QUESTION:", "WEBSITE CONTEXT:", "ANSWER:"]
                }
            },
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        if "response" in data:
            answer = clean_response(data["response"])
            lowered = answer.lower()

            forbidden_patterns = [
                "user question:",
                "website context:",
                "answer:",
                "to make tea",
                "you will need",
                "here are the steps",
            ]

            if any(pattern in lowered for pattern in forbidden_patterns):
                return WEBSITE_ONLY_FALLBACK

            return answer if answer else WEBSITE_ONLY_FALLBACK

        if "error" in data:
            return f"⚠️ Ollama Error: {data['error']}"

        return "⚠️ Model returned an unexpected response."

    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama connection failed. Make sure Ollama is running on http://localhost:11434"
    except requests.exceptions.Timeout:
        return "⚠️ Ollama took too long to respond."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ HTTP Error: {str(e)}"
    except Exception as e:
        return f"⚠️ Connection Error: {str(e)}"