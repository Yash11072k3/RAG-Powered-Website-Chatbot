import re
import streamlit as st
from scraper import scrape_website
from embeddings import split_text, create_embeddings, store_embeddings, hybrid_search
from llm import generate_answer, WEBSITE_ONLY_FALLBACK

st.set_page_config(page_title="RAG-Powered Website Chatbot", layout="wide")

FIXED_MODEL_NAME = "phi"


def init_session():
    defaults = {
        "index": None,
        "chunks": None,
        "messages": [],
        "loaded_url": "",
        "website_text": "",
        "website_keywords": set()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


STOPWORDS = {
    "what", "is", "are", "was", "were", "how", "why", "when", "where", "who",
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "this",
    "that", "these", "those", "about", "with", "from", "by", "it", "as", "be",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "tell", "me", "define", "explain", "article", "website", "page", "meaning"
}


def extract_keywords(text: str):
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def build_website_keyword_set(text: str):
    return set(extract_keywords(text))


def extract_question_keywords(question: str):
    return set(extract_keywords(question))


def keyword_overlap_count(question: str, text: str) -> int:
    q_keywords = set(extract_keywords(question))
    t_keywords = set(extract_keywords(text))
    return len(q_keywords.intersection(t_keywords))


def improve_query(question: str) -> str:
    q = question.lower().strip()

    if any(word in q for word in ["summar", "overview", "describe", "explain"]):
        return "main topic overview summary key points"

    if any(word in q for word in ["goal", "purpose", "aim", "objective"]):
        return question + " goal purpose objective"

    if any(word in q for word in ["risk", "danger", "dangerous", "harm", "safe", "safety", "threat"]):
        return question + " risks concerns harms safety"

    if any(word in q for word in ["history", "background", "origin", "started"]):
        return question + " history background origin development"

    if any(word in q for word in ["application", "applications", "use", "uses", "used"]):
        return question + " applications uses examples"

    if any(word in q for word in ["ethics", "ethical", "bias", "fairness"]):
        return question + " ethics ethical issues bias fairness"

    return question


def is_summary_question(question: str) -> bool:
    q = question.lower().strip()
    patterns = [
        "summar",
        "overview",
        "entire article",
        "full article",
        "whole article",
        "entire website",
        "whole website",
        "what is this article about",
        "what is this website about",
        "describe this page",
        "describe this website",
        "explain this article",
        "explain this page",
    ]
    return any(pattern in q for pattern in patterns)


def build_context(question: str, index, chunks):
    if is_summary_question(question):
        summary_results = hybrid_search(
            query=improve_query(question),
            index=index,
            chunks=chunks,
            top_k=8
        )
        context = "\n\n".join([r["text"] for r in summary_results]).strip()
        return context, summary_results

    raw_results = hybrid_search(
        query=question,
        index=index,
        chunks=chunks,
        top_k=4
    )

    improved_results = hybrid_search(
        query=improve_query(question),
        index=index,
        chunks=chunks,
        top_k=4
    )

    combined = []
    seen = set()
    for group in [raw_results, improved_results]:
        for item in group:
            if item["id"] not in seen:
                combined.append(item)
                seen.add(item["id"])

    combined.sort(key=lambda x: x["score"], reverse=True)

    chosen = combined[:4]
    context = "\n\n".join([r["text"] for r in chosen]).strip()

    if not context:
        return "", []

    return context, chosen


def is_question_relevant(question: str, index, chunks, website_keywords: set):
    if is_summary_question(question):
        return True, []

    q_keywords = extract_question_keywords(question)

    if not q_keywords:
        return False, []

    matched_keywords = q_keywords.intersection(website_keywords)

    # strict keyword rule
    if len(q_keywords) <= 2:
        if len(matched_keywords) < 1:
            return False, []
    else:
        if len(matched_keywords) < 2:
            return False, []

    results = hybrid_search(
        query=question,
        index=index,
        chunks=chunks,
        top_k=5
    )

    if not results:
        return False, []

    best_score = results[0]["score"]

    overlap_hits = 0
    for result in results:
        overlap_hits = max(overlap_hits, keyword_overlap_count(question, result["text"]))

    is_relevant = (
        best_score >= 0.22 or
        (best_score >= 0.18 and overlap_hits >= 1)
    )

    return is_relevant, results


init_session()

st.title("RAG-Powered Website Chatbot")
st.caption("Ask anything. Answers are grounded only in the loaded website.")


@st.cache_data(show_spinner=False)
def load_website_text(url: str) -> str:
    return scrape_website(url)


with st.sidebar:
    url = st.text_input("Enter Website URL", value=st.session_state.loaded_url)

    if st.button("Load Website", use_container_width=True):
        if not url.strip():
            st.warning("Please enter a website URL.")
        else:
            with st.spinner("Scraping and indexing website..."):
                text = load_website_text(url.strip())

                if text.startswith("Error:"):
                    st.error(text)
                else:
                    chunks = split_text(text, chunk_size=900, overlap=150)

                    if not chunks:
                        st.error("No valid chunks were created from the website content.")
                    else:
                        embeddings = create_embeddings(chunks)
                        index = store_embeddings(embeddings)
                        website_keywords = build_website_keyword_set(text)

                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.loaded_url = url.strip()
                        st.session_state.website_text = text
                        st.session_state.website_keywords = website_keywords
                        st.session_state.messages = []

                        st.success(
                            f"✅ Website loaded successfully! Total chunks: {len(chunks)} | "
                            f"Stored keywords: {len(website_keywords)}"
                        )

    if st.session_state.loaded_url:
        st.divider()
        st.write("**Loaded URL:**")
        st.caption(st.session_state.loaded_url)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if (
            msg["role"] == "assistant"
            and msg.get("is_relevant", False)
            and msg.get("source_previews")
        ):
            with st.expander("View source preview"):
                for preview in msg["source_previews"]:
                    st.markdown(f"**{preview['label']}**")
                    st.caption(preview["text"])


question = st.chat_input("Ask something about the website...")

if question:
    if st.session_state.index is None or st.session_state.chunks is None:
        st.warning("⚠️ Please load a website first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        relevant, _ = is_question_relevant(
            question=question,
            index=st.session_state.index,
            chunks=st.session_state.chunks,
            website_keywords=st.session_state.website_keywords
        )

        with st.chat_message("assistant"):
            previews = []

            if not relevant:
                answer = WEBSITE_ONLY_FALLBACK
            else:
                context, retrieved_chunks = build_context(
                    question=question,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks
                )

                if not context.strip():
                    answer = WEBSITE_ONLY_FALLBACK
                    relevant = False
                else:
                    with st.spinner("Searching..."):
                        answer = generate_answer(
                            context=context,
                            question=question,
                            model_name=FIXED_MODEL_NAME
                        )

                    if answer.strip() == WEBSITE_ONLY_FALLBACK:
                        relevant = False
                    else:
                        for i, item in enumerate(retrieved_chunks, start=1):
                            preview_text = item["text"][:350] + ("..." if len(item["text"]) > 350 else "")
                            previews.append({
                                "label": f"Reference {i}",
                                "text": preview_text
                            })

            st.write(answer)

            if relevant and previews:
                with st.expander("View source preview"):
                    for preview in previews:
                        st.markdown(f"**{preview['label']}**")
                        st.caption(preview["text"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "source_previews": previews if relevant else [],
            "is_relevant": relevant
        })
