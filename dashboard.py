from __future__ import annotations

import re

import streamlit as st

from memory import add_fact, list_facts, search_facts

st.set_page_config(page_title="Pet Memory Lab")
st.title("Pet Memory Lab")
st.write(
    "Add a fact, ask a question, and compare keyword and embedding search "
    "on the same memories."
)


def tokens(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 2]


def keyword_match(query: str, facts: list[dict]) -> list[dict]:
    """Rank facts by how many query words appear in the fact text.

    A score of 0 is not a match. Callers should say no memory was found
    only when every fact scores 0.
    """
    query_tokens = tokens(query)
    matches = []
    for fact in facts:
        fact_tokens = set(tokens(fact["fact"]))
        score = sum(1 for tok in query_tokens if tok in fact_tokens)
        if score > 0:
            matches.append(
                {"id": fact["id"], "fact": fact["fact"], "score": score}
            )
    matches.sort(key=lambda item: (-item["score"], str(item["id"])))
    return matches


def show_hit(hit: dict, score_line: str = "") -> None:
    st.write(hit["fact"])
    st.caption(f"ID: {hit['id']}")
    if score_line and hit.get("score") is not None:
        st.caption(score_line.format(score=hit["score"]))


def render_results(facts: list[dict], query: str) -> None:
    keyword_hits = keyword_match(query, facts) if facts else []
    embedding_error = None
    embedding_hit = None
    if facts:
        try:
            embedding_hit = search_facts(query)
        except Exception as exc:
            embedding_error = str(exc)

    keyword_col, embedding_col = st.columns(2)
    with keyword_col:
        st.subheader("Keyword")
        if not facts:
            st.info("No memories are saved yet.")
        elif not keyword_hits:
            st.info("No memory found")
        else:
            for hit in keyword_hits:
                show_hit(hit, "Overlapping words: {score}")
    with embedding_col:
        st.subheader("Embedding")
        if not facts:
            st.info("No memories are saved yet.")
        elif embedding_error:
            st.error(embedding_error)
        elif embedding_hit is None:
            st.info("No memories are saved yet.")
        else:
            show_hit(embedding_hit)
            st.caption("A nearest fact can still be irrelevant.")


st.subheader("Saved memories")
facts = list_facts()
if not facts:
    st.info("No memories are saved yet.")
else:
    for fact in facts:
        show_hit(fact)

st.subheader("Add a fact")
if st.session_state.get("saved_message"):
    st.success(st.session_state.saved_message)

with st.form("add_fact_form", clear_on_submit=True):
    fact_text = st.text_area("Fact", placeholder="My favorite drink is matcha.")
    save_clicked = st.form_submit_button("Save")

if save_clicked:
    if not fact_text or not fact_text.strip():
        st.warning("Enter a fact before saving.")
    else:
        saved = add_fact(fact_text.strip())
        st.session_state.saved_message = f"Saved fact. ID: {saved['id']}"
        st.rerun()

st.subheader("Ask a question")
with st.form("ask_form"):
    question = st.text_input("Question", placeholder="What do I like to drink?")
    search_clicked = st.form_submit_button("Search")

blank_question = bool(search_clicked and not (question or "").strip())
if blank_question:
    st.warning("Enter a question before searching.")
elif search_clicked:
    st.session_state.search_query = question.strip()

if st.session_state.get("search_query") and not blank_question:
    render_results(list_facts(), st.session_state.search_query)
