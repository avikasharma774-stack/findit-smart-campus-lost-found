import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date

st.set_page_config(page_title="FindIt | Campus Lost & Found", page_icon="🔎", layout="wide")

SAMPLE_DATA = [
    {"ID": 1, "Type": "Lost", "Item name": "Black water bottle", "Category": "Bottle",
     "Description": "Black Milton steel bottle with a silver cap. Medium size.",
     "Location": "College library", "Date": "2026-11-10"},
    {"ID": 2, "Type": "Found", "Item name": "Black steel bottle", "Category": "Bottle",
     "Description": "Found a black bottle with metallic lid near the library entrance.",
     "Location": "Library entrance", "Date": "2026-11-10"},
    {"ID": 3, "Type": "Lost", "Item name": "Blue pencil pouch", "Category": "Stationery",
     "Description": "Navy blue zip pouch with pens and a small ruler inside.",
     "Location": "Room 204", "Date": "2026-11-11"},
    {"ID": 4, "Type": "Found", "Item name": "Blue pouch", "Category": "Stationery",
     "Description": "Blue pencil case containing two pens, found in classroom 204.",
     "Location": "Classroom 204", "Date": "2026-11-11"},
    {"ID": 5, "Type": "Found", "Item name": "Wireless earbud case", "Category": "Electronics",
     "Description": "White charging case for wireless earbuds. No earbuds inside.",
     "Location": "Cafeteria", "Date": "2026-11-11"},
]

def make_text(row):
    """Combine useful fields so the model can compare more than just the item name."""
    return " ".join([
        str(row.get("Item name", "")),
        str(row.get("Category", "")),
        str(row.get("Description", "")),
        str(row.get("Location", "")),
    ])

def find_matches(query_row, candidate_rows):
    """Return candidates ranked by TF-IDF cosine similarity."""
    if not candidate_rows:
        return []

    all_texts = [make_text(query_row)] + [make_text(row) for row in candidate_rows]

    # TF-IDF turns words into numbers. Cosine similarity compares those numbers.
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(all_texts)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    # NumPy helps us find the index of the highest score.
    order = np.argsort(scores)[::-1]
    feature_names = vectorizer.get_feature_names_out()
    query_words = set(query_row.get("Description", "").lower().split())

    results = []
    for index in order:
        candidate = candidate_rows[int(index)]
        score = float(scores[int(index)])
        candidate_words = set(candidate.get("Description", "").lower().split())
        shared = sorted(query_words.intersection(candidate_words))
        results.append({
            "candidate": candidate,
            "score": score,
            "shared_words": shared[:8],
        })
    return results

if "items" not in st.session_state:
    st.session_state["items"] = pd.DataFrame(SAMPLE_DATA)

st.title("🔎 FindIt")
st.caption("Smart Campus Lost & Found — find likely matches between lost and found item reports.")
st.info("Prototype demo: sample reports are included. Similarity scores are suggestions, not proof that two reports refer to the same object.")

items = st.session_state["items"].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Total reports", len(items))
col2.metric("Lost reports", int((items["Type"] == "Lost").sum()))
col3.metric("Found reports", int((items["Type"] == "Found").sum()))

tab1, tab2, tab3 = st.tabs(["➕ Add a report", "✨ Find matches", "📋 All reports"])

with tab1:
    st.subheader("Report a lost or found item")
    with st.form("report_form", clear_on_submit=True):
        report_type = st.radio("I want to report", ["Lost", "Found"], horizontal=True)
        item_name = st.text_input("Item name", placeholder="e.g. Black water bottle")
        category = st.selectbox("Category", ["Bottle", "Electronics", "Stationery", "Bag", "Clothing", "ID / Cards", "Other"])
        description = st.text_area("Describe the item", placeholder="Colour, brand, marks, contents, or other identifying details...")
        location = st.text_input("Where was it lost/found?", placeholder="e.g. College library")
        report_date = st.date_input("Date", value=date.today())
        submitted = st.form_submit_button("Save report", type="primary")

        if submitted:
            if not item_name.strip() or not description.strip() or not location.strip():
                st.error("Please fill in the item name, description, and location.")
            else:
                new_id = int(st.session_state["items"]["ID"].max()) + 1 if not st.session_state["items"].empty else 1
                new_row = {
                    "ID": new_id, "Type": report_type, "Item name": item_name.strip(),
                    "Category": category, "Description": description.strip(),
                    "Location": location.strip(), "Date": str(report_date)
                }
                st.session_state["items"] = pd.concat(
                    [st.session_state["items"], pd.DataFrame([new_row])], ignore_index=True
                )
                st.success(f"Report #{new_id} saved! Go to Find matches to search for a possible match.")

with tab2:
    st.subheader("Search for a possible match")
    current = st.session_state["items"].copy()
    lost_rows = current[current["Type"] == "Lost"].to_dict("records")
    found_rows = current[current["Type"] == "Found"].to_dict("records")

    if not lost_rows or not found_rows:
        st.warning("Add at least one lost report and one found report to compare them.")
    else:
        choice_type = st.radio("I want to match a", ["Lost item", "Found item"], horizontal=True)
        if choice_type == "Lost item":
            options = {f'#{r["ID"]} — {r["Item name"]}': r for r in lost_rows}
            candidates = found_rows
        else:
            options = {f'#{r["ID"]} — {r["Item name"]}': r for r in found_rows}
            candidates = lost_rows

        selected_label = st.selectbox("Choose a report", list(options.keys()))
        query_row = options[selected_label]

        if st.button("Find possible matches", type="primary"):
            results = find_matches(query_row, candidates)
            if results:
                st.markdown("#### Best matches")
                for rank, result in enumerate(results[:3], start=1):
                    candidate = result["candidate"]
                    percentage = round(result["score"] * 100)
                    with st.container(border=True):
                        left, right = st.columns([3, 1])
                        with left:
                            st.markdown(f"**{rank}. {candidate['Item name']}**")
                            st.write(candidate["Description"])
                            st.caption(f"Category: {candidate['Category']}  |  Location: {candidate['Location']}  |  Date: {candidate['Date']}")
                            if result["shared_words"]:
                                st.caption("Some shared words: " + ", ".join(result["shared_words"]))
                        with right:
                            st.metric("Text similarity", f"{percentage}%")
                st.caption("The score measures similarity between the text fields. Always verify an item with its owner before returning it.")
            else:
                st.warning("No possible matches found.")

with tab3:
    st.subheader("All campus reports")
    current = st.session_state["items"].copy()
    type_filter = st.selectbox("Filter reports", ["All", "Lost", "Found"])
    if type_filter != "All":
        current = current[current["Type"] == type_filter]
    st.dataframe(current, use_container_width=True, hide_index=True)
    csv = st.session_state["items"].to_csv(index=False).encode("utf-8")
    st.download_button("Download reports as CSV", csv, "findit_reports.csv", "text/csv")

st.divider()
st.caption("Built as a learning prototype with Python, Pandas, NumPy, Scikit-learn and Streamlit.")
