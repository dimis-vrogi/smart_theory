from docx import Document
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DOC_PATH = "/content/Trapeza.docx" 
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

print(f" Φορτωμένες 'Ασκήσεις' : {len(df)}")


model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

df["exercise_text"] = df["exercise_text"].astype(str)

df["emb"] = model.encode(
    df["exercise_text"].tolist(),
    show_progress_bar=True
).tolist()


chapter_embeddings = {}

for chapter, group in df.groupby("chapter"):
    embs = np.array(group["emb"].tolist())
    chapter_embeddings[chapter] = embs.mean(axis=0)

print("✔ Trained chapters:", list(chapter_embeddings.keys()))


def predict_best_chapter(text):
    emb = model.encode([text])[0]

    best_chapter = None
    best_score = -1.0

    for chapter, chap_emb in chapter_embeddings.items():
        sim = cosine_similarity(
            emb.reshape(1, -1),
            chap_emb.reshape(1, -1)
        )[0][0]

        if sim > best_score:
            best_score = float(sim)
            best_chapter = chapter

    return best_chapter, best_score

print("Πληκτρολογήστε την άσκηση. Πατήστε ΈΞΟΔΟ για να σταματήσει το πρόγραμμα.\n")

while True:
    user_input = input("✍️ Exercise:\n")
    if user_input.strip().upper() == "ΈΞΟΔΟ":
        break

    chapter, score = predict_best_chapter(user_input)

    print("\n Πιθανό Κεφάλαιο:")
    print(f"• {chapter}  |  Score: {score:.3f}")
    print("-" * 50)
