import streamlit as st
from deepface import DeepFace
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import os
import zipfile

st.set_page_config(page_title="מערכת נוכחות חכמה", layout="wide")
st.title("📸 מערכת נוכחות חכמה")

ZIP_PATH = "My_Classmates_small.zip"
EXTRACT_PATH = "My_Classmates"
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = "My_Classmates/content/My_Classmates_small"
STUDENT_ROSTER = ['Maayan','Tomer','Roei','Zohar','Ilay']

# -------------------------
# טעינת embeddings של reference
# -------------------------
@st.cache_resource
def load_reference_embeddings():
    embeddings = {}
    for student in os.listdir(REFERENCE_DIR):
        student_path = os.path.join(REFERENCE_DIR, student)
        if os.path.isdir(student_path):
            student_embeddings = []
            for file in os.listdir(student_path):
                if file.lower().endswith((".jpg",".jpeg",".png")):
                    img_path = os.path.join(student_path, file)
                    try:
                        result = DeepFace.represent(
                            img_path=img_path,
                            model_name="Facenet512",
                            detector_backend="retinaface",  # ← חיתוך אוטומטי
                            enforce_detection=False
                        )
                        emb = np.array(result[0]["embedding"])
                        emb = emb / np.linalg.norm(emb)
                        student_embeddings.append(emb)
                    except Exception as e:
                        st.warning(f"שגיאה ב-{file}: {e}")
            if student_embeddings:
                embeddings[student] = student_embeddings
    return embeddings
reference_embeddings = load_reference_embeddings()
st.info(f"נמצאו {len(reference_embeddings)} תלמידים במאגר")

# -------------------------
# חיתוך פנים
# -------------------------
def extract_faces(image, confidence_threshold=0.7):
    import mediapipe as mp
    img_rgb = np.array(image.convert("RGB"))
    mp_face = mp.solutions.face_detection
    faces = []
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=confidence_threshold) as detector:
        results = detector.process(img_rgb)
        if not results.detections:
            return faces, img_rgb
        h, w = img_rgb.shape[:2]
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))
            pad_x = int(0.2 * (x2 - x1))
            pad_y = int(0.2 * (y2 - y1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            face = img_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_img = Image.fromarray(face).resize((160, 160))
            faces.append({"face": face_img, "box": (x1, y1, x2-x1, y2-y1)})
    return faces, img_rgb

# -------------------------
# cosine distance
# -------------------------
def cosine_distance(a, b):
    return 1 - np.dot(a, b)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("הגדרות")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.4)
    confidence = st.slider("Face Detection Confidence", 0.5, 1.0, 0.7)
    st.write("תלמידים בכיתה")
    for s in STUDENT_ROSTER:
        st.write(s)

st.subheader("העלי תמונת כיתה")
class_file = st.file_uploader("Upload class photo", type=["jpg","jpeg","png"])

if st.button("בדוק נוכחות"):
    if class_file is None:
        st.warning("יש להעלות תמונה")
        st.stop()

    class_image = Image.open(class_file)
    class_image = ImageOps.exif_transpose(class_image)

    faces, original_img_rgb = extract_faces(class_image, confidence)
    st.write(f"זוהו {len(faces)} פנים")

    present_students = {}
    recognized_faces = []

    for i, data in enumerate(faces):
        img = data["face"]
        box = data["box"]

        # DeepFace embedding על הפנים החתוכות
        try:
            img_array = np.array(img)
            result = DeepFace.represent(
                img_path=img_array,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False
            )
            emb = np.array(result[0]["embedding"])
            emb = emb / np.linalg.norm(emb)
        except Exception as e:
            st.warning(f"שגיאה בפנים {i+1}: {e}")
            continue

        # מרחק מינימלי לכל תלמיד
        avg_distances = {}
        for name, ref_embs in reference_embeddings.items():
            dists = [cosine_distance(emb, ref_emb) for ref_emb in ref_embs]
            avg_distances[name] = min(dists)

        best_name, best_dist = min(avg_distances.items(), key=lambda x: x[1])
        if best_dist > threshold:
            best_name = None

        # DEBUG
        st.write(f"--- פנים #{i+1} ---")
        st.image(img, width=100, caption=f"פנים {i+1}")
        for name, dist in sorted(avg_distances.items(), key=lambda x: x[1]):
            st.write(f"{name}: {dist:.4f}")
        st.write(f"זוהה כ: {best_name} (מרחק={best_dist:.4f})")

        if best_name and best_name not in present_students:
            present_students[best_name] = img
            recognized_faces.append({"name": best_name, "box": box})

    # ציור תיבות
    img_draw = Image.fromarray(original_img_rgb)
    draw = ImageDraw.Draw(img_draw)
    for face in recognized_faces:
        x, y, w, h = face["box"]
        name = face["name"]
        draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=3)
        draw.text((x, y-20), name, fill=(0,255,0))

    st.subheader("תוצאת זיהוי")
    st.image(img_draw, use_column_width=True)

    missing_students = [s for s in STUDENT_ROSTER if s not in present_students]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.header(f"✅ נוכחים ({len(present_students)})")
        cols = st.columns(3)
        for i, (name, img) in enumerate(present_students.items()):
            with cols[i % 3]:
                st.write(f"**{name}**")
                st.image(img, width=90)

    with col2:
        st.header(f"❌ חסרים ({len(missing_students)})")
        if missing_students:
            for s in missing_students:
                st.write(s)
        else:
            st.success("כולם נוכחים")
