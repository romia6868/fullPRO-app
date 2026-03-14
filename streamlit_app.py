import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image, ImageOps
import numpy as np
import os
import zipfile
import cv2
from retinaface import RetinaFace

st.set_page_config(page_title="מערכת נוכחות חכמה", layout="wide")
st.title("📸 מערכת נוכחות חכמה")

ZIP_PATH = "My_Classmates_small.zip"
EXTRACT_PATH = "My_Classmates"
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = "My_Classmates/content/My_Classmates_small"
STUDENT_ROSTER = ['Maayan','Tomer','Roei','Zohar','Ilay']

class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def build_pro_embedding():
    base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(128),
        L2Normalize()
    ])
    return model

@st.cache_resource
def load_model():
    model = build_pro_embedding()
    model(np.zeros((1,224,224,3)))
    model.load_weights("face_encoder.weights.h5", by_name=True, skip_mismatch=True)
    return model

model = load_model()
st.success("המודל נטען בהצלחה")

def preprocess_image(img):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)


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
                    img = Image.open(img_path)
                    img = ImageOps.exif_transpose(img)
                    
                    # נריץ RetinaFace גם על תמונות ה-reference
                    faces, _ = extract_faces(img, 0.5)
                    
                    if faces:
                        # אם זוהתה פנים – נשתמש בה
                        face_img = faces[0]["face"]
                    else:
                        # אם לא זוהתה – נשתמש בתמונה המקורית כ-fallback
                        face_img = img
                    
                    emb = model.predict(preprocess_image(face_img), verbose=0)[0]
                    emb = emb / np.linalg.norm(emb)
                    student_embeddings.append(emb)
            if student_embeddings:
                embeddings[student] = student_embeddings
    return embeddings

reference_embeddings = load_reference_embeddings()
st.info(f"נמצאו {len(reference_embeddings)} תלמידים במאגר")

def extract_faces(image, confidence_threshold=0.90):
    img_rgb = np.array(image.convert("RGB"))
    detections = RetinaFace.detect_faces(img_rgb)
    faces = []
    if not isinstance(detections, dict):
        return faces, img_rgb
    for key, det in detections.items():
        score = det.get("score", 1.0)
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = det["facial_area"]
        w = x2 - x1
        h = y2 - y1
        pad_x = int(0.2 * w)
        pad_y = int(0.2 * h)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_rgb.shape[1], x2 + pad_x)
        y2 = min(img_rgb.shape[0], y2 + pad_y)
        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face_resized = cv2.resize(face, (224, 224))
        face_img = Image.fromarray(face_resized)
        faces.append({"face": face_img, "box": (x1, y1, x2-x1, y2-y1), "score": score})
    return faces, img_rgb

with st.sidebar:
    st.header("הגדרות")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.14)
    confidence = st.slider("Face Detection Confidence", 0.5, 1.0, 0.90)
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
        
        emb = model.predict(preprocess_image(img), verbose=0)[0]
        emb = emb / np.linalg.norm(emb)
        
        distances = []
        for name, ref_embs in reference_embeddings.items():
            for ref_emb in ref_embs:
                dist = cosine_distance(emb, ref_emb)
                distances.append((name, dist))
        
        distances.sort(key=lambda x: x[1])
        
        # DEBUG
        st.write(f"--- פנים #{i+1} ---")
        st.image(img, width=100, caption=f"פנים {i+1}")
        for name, dist in distances[:5]:
            st.write(f"{name}: {dist:.4f}")
        
        top_matches = distances[:5]
        votes = {}
        for name, dist in top_matches:
            if dist < threshold:
                votes[name] = votes.get(name, 0) + 1
        
        best_name = max(votes, key=votes.get) if votes else None
        st.write(f"זוהה כ: {best_name} (threshold={threshold})")
        
        if best_name and best_name not in present_students:
            present_students[best_name] = img
            recognized_faces.append({"name": best_name, "box": box})

    img_draw = original_img_rgb.copy()
    for face in recognized_faces:
        x, y, w, h = face["box"]
        name = face["name"]
        cv2.rectangle(img_draw, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img_draw, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

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
