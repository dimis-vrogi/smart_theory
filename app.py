import streamlit as st
from docx import Document
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Smart Theory Matcher",
    layout="centered"
)

st.title(" Smart Theory Matcher")
st.write("Επικόλλησε μια άσκηση και θα σου επιστρέψω το πιθανότερο κεφάλαιο.")


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"  # 🔥 ελαφρύ & γρήγορο
    )

model = load_model()


@st.cache_data
def load_dataset():
    DOC_PATH = "Trapeza.docx"
    doc = Document(DOC_PATH)

    rows = []
    current_chapter = None
    current_exercise = None
    buffer = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        if text.startswith("ΚΕΦΑΛΑΙΟ:"):
            current_chapter = text.replace("ΚΕΦΑΛΑΙΟ:", "").strip()

        elif text.startswith("ΘΕΜΑ"):
            if current_exercise and buffer and current_chapter:
                chapters = [c.strip() for c in current_chapter.split(",")]
                for ch in chapters:
                    rows.append({
                        "exercise_text": " ".join(buffer),
                        "chapter": ch
                    })
            current_exercise = text
            buffer = []

        else:
            buffer.append(text)

    if current_exercise and buffer and current_chapter:
        chapters = [c.strip() for c in current_chapter.split(",")]
        for ch in chapters:
            rows.append({
                "exercise_text": " ".join(buffer),
                "chapter": ch
            })

    df = pd.DataFrame(rows)
    df = df[df["chapter"] != "ΑΓΝΩΣΤΟ ΚΕΦΑΛΑΙΟ"].reset_index(drop=True)
    return df

df = load_dataset()

st.success(f"Φορτώθηκαν {len(df)} ασκήσεις")


@st.cache_data
def compute_embeddings(texts):
    return model.encode(texts)

df["emb"] = list(compute_embeddings(df["exercise_text"].tolist()))


chapter_embeddings = {}

for chapter, group in df.groupby("chapter"):
    embs = np.vstack(group["emb"].values)
    chapter_embeddings[chapter] = embs.mean(axis=0)


def predict_best_chapter(text):
    query_emb = model.encode([text])[0]

    best_chapter = None
    best_score = -1.0

    for chapter, chap_emb in chapter_embeddings.items():
        sim = cosine_similarity(
            query_emb.reshape(1, -1),
            chap_emb.reshape(1, -1)
        )[0][0]

        if sim > best_score:
            best_score = float(sim)
            best_chapter = chapter

    return best_chapter, best_score


user_text = st.text_area(
    " Επικόλλησε την άσκηση εδώ:",
    height=180
)

if st.button(" Αντιστοίχισε Κεφάλαιο"):
    if user_text.strip():
        chapter, score = predict_best_chapter(user_text)

        st.subheader(" Πιθανότερο Κεφάλαιο")
        st.write(f"**{chapter}**")
        st.caption(f"Similarity score: {score:.3f}")
    else:
        st.warning("Γράψε πρώτα μια άσκηση.")
