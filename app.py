import streamlit as st
from scraper import scrape_website
from embeddings import split_text, create_embeddings, store_embeddings, hybrid_search
from llm import generate_answer, WEBSITE_ONLY_FALLBACK

st.set_page_config(page_title="RAG Powered Website Chatbot", layout="wide")

FIXED_MODEL_NAME = "phi"


def init_session():
    defaults = {
        "index": None,
        "chunks": None,
        "messages": [],
        "loaded_url": "",
        "website_text": ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()

st.title("RAG Powered Website Chatbot")
st.caption("Ask anything. Answers are grounded only in the loaded website.")


@st.cache_data(show_spinner=False)
def load_website_text(url: str) -> str:
    return scrape_website(url)


def improve_query(question: str) -> str:
    q = question.lower().strip()

    if any(word in q for word in ["summar", "overview", "describe", "explain"]):
        return "main topic overview key points summary"

    if "what is" in q or "about" in q:
        return "main topic explanation definition"

    if any(word in q for word in ["danger", "dangerous", "risk", "harm", "threat", "safe", "safety"]):
        return "risks dangers concerns safety issues"

    if "application" in q or "applications" in q:
        return "applications uses examples"

    if "history" in q:
        return "history background development"

    if "ethics" in q or "ethical" in q:
        return "ethics concerns issues"

    return question


def is_summary_question(question: str) -> bool:
    q = question.lower().strip()
    return any(word in q for word in [
        "summar",
        "overview",
        "entire article",
        "full article",
        "whole article",
        "whole website",
        "entire website",
        "describe this page",
        "describe this website",
        "explain this page",
        "explain this article",
        "what is this article about",
        "what is this website about"
    ])


def build_context(question: str, index, chunks):
    if is_summary_question(question):
        total = len(chunks)

        if total <= 10:
            selected_ids = list(range(len(chunks)))
            selected_chunks = chunks
        else:
            positions = [
                int(total * 0.05),
                int(total * 0.15),
                int(total * 0.25),
                int(total * 0.35),
                int(total * 0.45),
                int(total * 0.55),
                int(total * 0.65),
                int(total * 0.75),
                int(total * 0.85),
                max(int(total * 0.95) - 1, 0)
            ]

            seen = set()
            selected_ids = []
            for pos in positions:
                pos = max(0, min(pos, total - 1))
                if pos not in seen:
                    selected_ids.append(pos)
                    seen.add(pos)

            selected_chunks = [chunks[i] for i in selected_ids]

        context = "\n\n".join(selected_chunks).strip()
        retrieved_chunks = [
            {"id": chunk_id, "text": chunks[chunk_id], "score": 0.0}
            for chunk_id in selected_ids
        ]
        return context, retrieved_chunks

    search_query = improve_query(question)

    results = hybrid_search(
        query=search_query,
        index=index,
        chunks=chunks,
        top_k=6
    )

    chosen = results[:4]
    context = "\n\n".join([r["text"] for r in chosen]).strip()

    if not context:
        return "", []

    return context, chosen


def is_question_relevant(question: str, index, chunks):
    if is_summary_question(question):
        return True, []

    # Use original question for relevance
    raw_results = hybrid_search(
        query=question,
        index=index,
        chunks=chunks,
        top_k=3
    )

    # Use improved query for recall
    improved_results = hybrid_search(
        query=improve_query(question),
        index=index,
        chunks=chunks,
        top_k=3
    )

    combined = []
    seen_ids = set()

    for group in [raw_results, improved_results]:
        for item in group:
            if item["id"] not in seen_ids:
                combined.append(item)
                seen_ids.add(item["id"])

    if not combined:
        return False, []

    best_score = combined[0]["score"]

    question_words = [w for w in question.lower().split() if len(w) > 3]
    lexical_match_found = False

    for result in combined:
        chunk_text = result["text"].lower()
        if any(word in chunk_text for word in question_words):
            lexical_match_found = True
            break

    is_relevant = best_score >= 0.28 or (best_score >= 0.20 and lexical_match_found)

    return is_relevant, combined


with st.sidebar:
    st.header("⚙️ Settings")

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

                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.loaded_url = url.strip()
                        st.session_state.website_text = text
                        st.session_state.messages = []

                        st.success(f"✅ Website loaded successfully! Total chunks: {len(chunks)}")

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
            chunks=st.session_state.chunks
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