import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image, ImageOps
import numpy as np
import os
import zipfile
import cv2
from mtcnn import MTCNN

# -------------------------
# הגדרות דף
# -------------------------
st.set_page_config(page_title="מערכת נוכחות חכמה", layout="wide")
st.title("📸 מערכת נוכחות חכמה")

# -------------------------
# חילוץ מאגר התמונות
# -------------------------
ZIP_PATH = "My_Classmates_small.zip"
EXTRACT_PATH = "My_Classmates"

if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = "My_Classmates/content/My_Classmates_small"

# -------------------------
# רשימת תלמידים
# -------------------------
STUDENT_ROSTER = ['Maayan','Tomer','Roei','Zohar','Ilay']

# -------------------------
# שכבת נרמול
# -------------------------
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

# -------------------------
# בניית מודל
# -------------------------
def build_pro_embedding():

    base_model = MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )

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

# -------------------------
# טעינת מודל
# -------------------------
@st.cache_resource
def load_model():

    model = build_pro_embedding()
    model(np.zeros((1,224,224,3)))

    model.load_weights(
        "face_encoder.weights.h5",
        by_name=True,
        skip_mismatch=True
    )

    return model

model = load_model()
st.success("המודל נטען בהצלחה")

# -------------------------
# preprocessing
# -------------------------
def preprocess_image(img):

    img = img.convert("RGB").resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# -------------------------
# cosine distance
# -------------------------
def cosine_distance(a,b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return 1 - np.dot(a,b)

# -------------------------
# טעינת embeddings
# -------------------------
@st.cache_data
def load_reference_embeddings():

    embeddings = {}

    for student in os.listdir(REFERENCE_DIR):

        student_path = os.path.join(REFERENCE_DIR, student)

        if os.path.isdir(student_path):

            student_embeddings = []

            for file in os.listdir(student_path):

                if file.lower().endswith((".jpg",".jpeg",".png")):

                    img_path = os.path.join(student_path,file)

                    img = Image.open(img_path)
                    img = ImageOps.exif_transpose(img)

                    emb = model.predict(
                        preprocess_image(img),
                        verbose=0
                    )[0]

                    emb = emb / np.linalg.norm(emb)

                    student_embeddings.append(emb)

            if student_embeddings:
                embeddings[student] = student_embeddings

    return embeddings

reference_embeddings = load_reference_embeddings()

st.info(f"נמצאו {len(reference_embeddings)} תלמידים במאגר")

# -------------------------
# גלאי פנים
# -------------------------
@st.cache_resource
def load_face_detector():
    return MTCNN()

face_detector = load_face_detector()

# -------------------------
# חיתוך פנים
# -------------------------
def extract_faces(image):

    image = image.convert("RGB")
    img = np.array(image)

    detections = face_detector.detect_faces(img)

    faces = []

    for det in detections:

        x,y,w,h = det["box"]

        x = max(0,x)
        y = max(0,y)

        pad = int(0.4*w)

        x = max(0,x-pad)
        y = max(0,y-pad)

        w = w + 2*pad
        h = h + 2*pad

        face = img[y:y+h , x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face,(224,224))

        face_img = Image.fromarray(face)

        faces.append({
            "face":face_img,
            "box":(x,y,w,h)
        })

    return faces,img

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:

    st.header("הגדרות")

    threshold = st.slider(
        "Similarity Threshold",
        0.0,
        1.0,
        0.14
    )

    st.write("תלמידים בכיתה")

    for s in STUDENT_ROSTER:
        st.write(s)

# -------------------------
# העלאת תמונה
# -------------------------
st.subheader("העלי תמונת כיתה")

class_file = st.file_uploader(
    "Upload class photo",
    type=["jpg","jpeg","png"]
)

# -------------------------
# זיהוי
# -------------------------
if st.button("בדוק נוכחות"):

    if class_file is None:
        st.warning("יש להעלות תמונה")
        st.stop()

    class_image = Image.open(class_file)
    class_image = ImageOps.exif_transpose(class_image)

    faces, original_img = extract_faces(class_image)

    st.write(f"זוהו {len(faces)} פנים")

    present_students = {}
    recognized_faces = []

    for data in faces:

        img = data["face"]
        box = data["box"]

        emb = model.predict(
            preprocess_image(img),
            verbose=0
        )[0]

        emb = emb / np.linalg.norm(emb)

        distances = []

        for name, ref_embs in reference_embeddings.items():

            for ref_emb in ref_embs:

                dist = cosine_distance(emb, ref_emb)

                distances.append((name, dist))

        distances.sort(key=lambda x: x[1])

        top_matches = distances[:5]

        votes = {}

        for name, dist in top_matches:

            if dist < threshold:
                votes[name] = votes.get(name,0) + 1

        if votes:
            best_name = max(votes, key=votes.get)
        else:
            best_name = None

        if best_name and best_name not in present_students:
            present_students[best_name] = img

        recognized_faces.append({
            "name":best_name,
            "box":box
        })

    img_draw = original_img.copy()

    for face in recognized_faces:

        x,y,w,h = face["box"]

        name = face["name"] if face["name"] else "Unknown"

        cv2.rectangle(img_draw,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            img_draw,
            name,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    st.subheader("תוצאת זיהוי")

    st.image(
        cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
        use_column_width=True
    )

    missing_students = [
        s for s in STUDENT_ROSTER
        if s not in present_students
    ]

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        st.header(f"✅ נוכחים ({len(present_students)})")

        cols = st.columns(3)

        for i,(name,img) in enumerate(present_students.items()):

            with cols[i % 3]:
                st.write(f"**{name}**")
                st.image(img,width=90)

    with col2:

        st.header(f"❌ חסרים ({len(missing_students)})")

        if missing_students:
            for s in missing_students:
                st.write(s)
        else:
            st.success("כולם נוכחים")
