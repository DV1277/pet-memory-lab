from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from memory import add_fact, list_facts, search_facts, update_fact

RESULTS_PATH = Path(__file__).resolve().parent / "retrieval_results.csv"
OVERLAP_WARNING = 0.8
LINK_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "was",
    "are",
    "her",
    "his",
    "she",
    "they",
    "you",
    "your",
    "but",
    "not",
    "its",
    "our",
    "who",
    "what",
    "when",
    "where",
    "how",
}

PAW_SVG = """
<svg width="56" height="56" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <ellipse cx="33" cy="42" rx="13" ry="11" fill="#E07A5F"/>
  <circle cx="15" cy="24" r="6" fill="#E07A5F"/>
  <circle cx="27" cy="15" r="6.5" fill="#E07A5F"/>
  <circle cx="41" cy="15" r="6.5" fill="#E07A5F"/>
  <circle cx="52" cy="25" r="5.5" fill="#E07A5F"/>
</svg>
""".strip()
HEADER_INTRO = (
    "Save a memory, update one that changed, and compare keyword overlap "
    "with embedding distance on the same memories."
)


def make_paw_icon() -> Image.Image:
    """Terracotta paw used as the browser tab icon."""
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    color = (224, 122, 95, 255)
    draw.ellipse((18, 30, 50, 58), fill=color)
    draw.ellipse((6, 16, 22, 32), fill=color)
    draw.ellipse((20, 6, 36, 22), fill=color)
    draw.ellipse((36, 6, 52, 22), fill=color)
    draw.ellipse((48, 18, 62, 32), fill=color)
    return image


def memory_count_label(facts: list[dict] | None) -> str:
    if facts is None:
        return "Memories unavailable"
    count = len(facts)
    noun = "memory" if count == 1 else "memories"
    return f"{count} {noun} saved"


def render_header(facts: list[dict] | None) -> None:
    label = memory_count_label(facts)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:18px 22px;border-radius:16px;background:#D7EDD4;border:1px solid #C3DCC0;">
          <div style="flex:0 0 auto;line-height:0;">{PAW_SVG}</div>
          <div style="flex:1;min-width:220px;">
            <div style="font-size:2rem;font-weight:700;color:#3D2C29;line-height:1.15;">Pet Memory Lab</div>
            <div style="margin-top:6px;color:#3D2C29;font-size:1rem;">{HEADER_INTRO}</div>
          </div>
          <div style="flex:0 0 auto;background:#81B29A;color:#ffffff;border-radius:999px;padding:8px 14px;font-weight:600;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Pet Memory Lab",
    page_icon=make_paw_icon(),
    layout="wide",
)


def tokens(text: str) -> list[str]:
    """Searchable words and numbers. Single letters are ignored; digits stay."""
    found = re.findall(r"[^\W_]+", str(text).casefold())
    return [tok for tok in found if len(tok) >= 2 or tok.isdigit()]


def normalized_key(text: str) -> str:
    return " ".join(sorted(set(tokens(text))))


def keyword_match(query: str, facts: list[dict]) -> list[dict]:
    """Rank facts by how many query words appear in the fact text.

    Repeated query words count once. A score of 0 is not a match.
    Identical wording is grouped so duplicate saves stay visible.
    """
    query_tokens = set(tokens(query))
    matches = []
    grouped: dict[str, dict] = {}
    for fact in facts:
        fact_text = str(fact.get("fact", ""))
        fact_tokens = set(tokens(fact_text))
        score = len(query_tokens & fact_tokens)
        if score <= 0:
            continue
        hit = {
            "id": fact.get("id", ""),
            "fact": fact_text,
            "score": score,
            "also_ids": [],
        }
        key = normalized_key(fact_text) or str(hit["id"])
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = hit
            matches.append(hit)
        else:
            existing["also_ids"].append(hit["id"])
    matches.sort(key=lambda item: (-item["score"], str(item["id"])))
    return matches


def overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(tokens(left))
    right_tokens = set(tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def closest_overlap(text: str, facts: list[dict], skip_id: str | None = None) -> dict | None:
    best = None
    best_score = 0.0
    for fact in facts:
        if skip_id is not None and str(fact.get("id")) == str(skip_id):
            continue
        score = overlap_ratio(text, str(fact.get("fact", "")))
        if score > best_score:
            best = fact
            best_score = score
    if best is None or best_score < OVERLAP_WARNING:
        return None
    return {"fact": best, "score": best_score}


def duplicate_of(text: str, facts: list[dict], skip_id: str | None = None) -> dict | None:
    key = normalized_key(text)
    if not key:
        return None
    for fact in facts:
        if skip_id is not None and str(fact.get("id")) == str(skip_id):
            continue
        if normalized_key(str(fact.get("fact", ""))) == key:
            return fact
    return None


def duplicate_groups(facts: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for fact in facts:
        key = normalized_key(str(fact.get("fact", "")))
        if not key:
            continue
        groups.setdefault(key, []).append(fact)
    return [group for group in groups.values() if len(group) > 1]


def connected_memories(facts: list[dict]) -> list[tuple[str, list[dict]]]:
    """Words shared by at least two different memories."""
    buckets: dict[str, list[dict]] = {}
    for fact in facts:
        seen = set(tokens(str(fact.get("fact", ""))))
        for tok in seen:
            if len(tok) < 3 or tok.isdigit() or tok in LINK_STOPWORDS:
                continue
            buckets.setdefault(tok, []).append(fact)
    links = [(word, items) for word, items in buckets.items() if len(items) >= 2]
    links.sort(key=lambda item: (-len(item[1]), item[0]))
    return links


def short_label(hit: dict) -> str:
    text = " ".join(str(hit.get("fact", "")).split()) or "(empty)"
    if len(text) > 42:
        text = text[:39] + "..."
    prefix = str(hit.get("id", ""))[:8]
    return f"{text} ({prefix})" if prefix else text


def format_distance(value: object) -> str | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return f"{number:.3f}"


def show_hit(hit: dict, score_line: str = "") -> None:
    with st.container(border=True):
        fact = str(hit.get("fact", "")).strip()
        st.write(fact or "(empty memory)")
        st.caption(f"ID: {hit.get('id', '')}")
        also_ids = [str(item) for item in hit.get("also_ids", []) if item]
        if also_ids:
            st.caption("Same wording is also saved as: " + ", ".join(also_ids))
        if score_line and hit.get("score") is not None:
            st.caption(score_line.format(score=hit["score"]))
        distance = format_distance(hit.get("distance"))
        if distance is not None:
            st.caption(f"Distance: {distance} (smaller is closer)")


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
        elif not tokens(query):
            st.info("This question has no searchable words or numbers.")
        elif not keyword_hits:
            st.info("No memory found")
        else:
            match_count = len(keyword_hits)
            if match_count == 1:
                st.caption("1 memory shares words with this question.")
            else:
                st.caption(
                    f"{match_count} memories share words with this question, "
                    "ranked by overlap."
                )
            if (
                len(keyword_hits) > 1
                and keyword_hits[0]["score"] == keyword_hits[1]["score"]
            ):
                st.caption("The top memories tie. The lower ID is listed first.")
            chart = pd.DataFrame(
                {
                    "Memory": [short_label(hit) for hit in keyword_hits[:12]],
                    "Overlapping words": [hit["score"] for hit in keyword_hits[:12]],
                }
            )
            st.bar_chart(
                chart,
                x="Memory",
                y="Overlapping words",
                horizontal=True,
                sort=False,
                color="#81B29A",
            )
            if len(keyword_hits) > 12:
                st.caption("The chart shows the 12 strongest keyword matches.")
            for hit in keyword_hits:
                show_hit(hit, "Overlapping words: {score}")
    with embedding_col:
        st.subheader("Embedding")
        if not facts:
            st.info("No memories are saved yet.")
        elif embedding_error:
            st.error(embedding_error)
        elif not embedding_hit:
            st.info("Embedding search did not return a memory.")
        else:
            distance = format_distance(embedding_hit.get("distance"))
            if distance is not None:
                st.metric("Distance", distance, help="Smaller means a closer match.")
            show_hit(embedding_hit)
            st.caption("The nearest memory can still be off topic.")
            if keyword_hits:
                same = str(embedding_hit.get("id")) == str(keyword_hits[0].get("id"))
                if same:
                    st.caption("Keyword and embedding agree on the top memory.")
                else:
                    st.caption("Keyword and embedding picked different top memories.")


def _passed(row: dict[str, str]) -> bool | None:
    flag = (row.get("pass") or row.get("passed") or row.get("result") or "").strip().casefold()
    if flag in {"pass", "passed", "true", "yes", "1"}:
        return True
    if flag in {"fail", "failed", "false", "no", "0"}:
        return False
    expected = (row.get("expected_id") or row.get("expected") or "").strip()
    actual = (row.get("actual_id") or row.get("actual") or "").strip()
    if not expected and not actual:
        return None
    return expected == actual


def load_accuracy(path: Path) -> dict | None:
    """Read recorded retrieval checks, or None when no results file exists."""
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [name.strip() for name in (reader.fieldnames or [])]
            rows = []
            for raw in reader:
                row = {
                    (key or "").strip(): (value or "").strip()
                    for key, value in raw.items()
                }
                if any(row.values()):
                    rows.append(row)
    except OSError as exc:
        return {"error": f"Could not read retrieval results: {exc}"}

    needed = {"question", "expected_id", "actual_id"}
    # Accept the shorter expected/actual names as a pair.
    has_ids = needed.issubset(set(fieldnames)) or {"expected", "actual"}.issubset(
        set(fieldnames)
    )
    if "question" not in fieldnames or not has_ids:
        return {
            "error": (
                "retrieval_results.csv needs question, expected_id, and actual_id columns."
            )
        }
    checked = []
    for row in rows:
        outcome = _passed(row)
        if outcome is None:
            continue
        checked.append(
            {
                "Question": row.get("question", ""),
                "Expected ID": row.get("expected_id") or row.get("expected", ""),
                "Actual ID": row.get("actual_id") or row.get("actual", ""),
                "Result": "Pass" if outcome else "Fail",
            }
        )
    return {"rows": checked}


def render_accuracy() -> None:
    st.subheader("Retrieval accuracy")
    loaded = load_accuracy(RESULTS_PATH)
    if loaded is None:
        st.info(
            "Accuracy will appear here after test results are saved in "
            "retrieval_results.csv with question, expected_id, and actual_id."
        )
        return
    if loaded.get("error"):
        st.warning(loaded["error"])
        return
    rows = loaded["rows"]
    if not rows:
        st.info("The retrieval results file has no scored questions yet.")
        return
    passed = sum(1 for row in rows if row["Result"] == "Pass")
    st.metric("Retrieval accuracy", f"{passed / len(rows):.0%}", f"{passed}/{len(rows)} passed")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


st.session_state.setdefault("saved_message", "")
st.session_state.setdefault("search_query", "")
st.session_state.setdefault("search_error", "")

try:
    facts = list_facts()
except Exception as exc:
    facts = None
    load_error = exc
else:
    load_error = None

render_header(facts)

if load_error is not None:
    st.error(f"Saved memories could not be loaded: {load_error}")

if st.session_state.saved_message:
    st.success(st.session_state.saved_message)

save_clicked = False
update_clicked = False
add_col, update_col = st.columns(2)
with add_col:
    with st.container(border=True):
        st.subheader("Add a fact")
        with st.form("add_fact_form"):
            fact_text = st.text_area("Fact", placeholder="My favorite drink is matcha.")
            save_distinct = st.checkbox(
                "Save as a distinct fact even if the wording overlaps another memory"
            )
            save_clicked = st.form_submit_button(
                "Save", disabled=facts is None, type="primary"
            )

with update_col:
    with st.container(border=True):
        st.subheader("Update a fact")
        if not facts:
            st.info("Save a memory before updating one.")
        else:
            facts_by_id = {str(fact.get("id", "")): fact for fact in facts}
            selected_id = st.selectbox(
                "Memory to update",
                options=list(facts_by_id),
                format_func=lambda fact_id: short_label(facts_by_id[fact_id]),
            )
            selected = facts_by_id[selected_id]
            with st.form("update_fact_form"):
                updated_text = st.text_area(
                    "Updated fact",
                    value=str(selected.get("fact", "")),
                    key=f"update_text_{selected_id}",
                )
                update_clicked = st.form_submit_button("Update", type="primary")

if facts is not None and save_clicked:
    cleaned = (fact_text or "").strip()
    if not cleaned:
        st.warning("Enter a fact before saving.")
    elif not tokens(cleaned):
        st.warning("Enter a fact that includes at least one word or number.")
    else:
        existing = duplicate_of(cleaned, facts)
        similar = closest_overlap(cleaned, facts)
        if existing:
            st.warning(
                "This memory is already saved. "
                f"ID: {existing.get('id')}. Update that memory instead of saving a copy."
            )
        elif similar and not save_distinct:
            st.warning(
                "This overlaps an existing memory "
                f"(ID: {similar['fact'].get('id')}). "
                "Update that memory if the detail changed. "
                "To keep both, check the distinct-fact box and save again."
            )
        else:
            try:
                saved = add_fact(cleaned)
            except Exception as exc:
                st.error(f"Could not save that fact: {exc}")
            else:
                st.session_state.saved_message = f"Saved fact. ID: {saved['id']}"
                st.rerun()

if facts and update_clicked:
    cleaned = (updated_text or "").strip()
    if not cleaned:
        st.warning("Enter a fact before updating.")
    elif not tokens(cleaned):
        st.warning("Enter a fact that includes at least one word or number.")
    elif normalized_key(cleaned) == normalized_key(str(selected.get("fact", ""))):
        st.info("That memory already says this.")
    else:
        existing = duplicate_of(cleaned, facts, skip_id=selected_id)
        if existing:
            st.warning(
                "Another memory already uses this wording. "
                f"ID: {existing.get('id')}."
            )
        else:
            try:
                saved = update_fact(selected_id, cleaned)
            except Exception as exc:
                st.error(f"Could not update that fact: {exc}")
            else:
                if saved is None:
                    st.error("That memory is no longer saved. Refresh and choose it again.")
                else:
                    st.session_state.saved_message = f"Updated fact. ID: {saved['id']}"
                    st.rerun()

st.subheader("Saved memories")
if facts is None:
    st.info("Saved memories are unavailable until the list can be loaded.")
elif not facts:
    st.info("No memories are saved yet.")
else:
    copies = duplicate_groups(facts)
    if copies:
        repeated = sum(len(group) for group in copies)
        st.warning(
            f"{repeated} saved memories reuse the same wording "
            f"across {len(copies)} group{'s' if len(copies) != 1 else ''}. "
            "Update one of them instead of saving another copy."
        )
    for fact in sorted(facts, key=lambda item: str(item.get("id", ""))):
        show_hit(fact)
    links = connected_memories(facts)
    with st.expander("Connected memories"):
        if not links:
            st.info("No memories share a word yet.")
        else:
            st.caption("Memories that share a word. This is a reading aid, not a new search.")
            for word, items in links[:12]:
                labels = ", ".join(short_label(item) for item in items[:6])
                st.write(f"**{word}** — {labels}")

st.subheader("Ask a question")
with st.form("ask_form"):
    question = st.text_input("Question", placeholder="What do I like to drink?")
    search_clicked = st.form_submit_button(
        "Search", disabled=facts is None, type="primary"
    )

if search_clicked:
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        st.session_state.search_error = "Enter a question before searching."
        st.session_state.search_query = ""
    elif not tokens(cleaned_question):
        st.session_state.search_error = (
            "Enter a question that includes at least one word or number."
        )
        st.session_state.search_query = ""
    else:
        st.session_state.search_error = ""
        st.session_state.search_query = cleaned_question

if st.session_state.search_error:
    st.warning(st.session_state.search_error)
elif facts and st.session_state.search_query:
    render_results(facts, st.session_state.search_query)

render_accuracy()
