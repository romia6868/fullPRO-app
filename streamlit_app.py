import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
# חילוץ מאגר תמונות
# -------------------------

ZIP_PATH = "My_Classmates_small.zip"
EXTRACT_PATH = "My_Classmates"

if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = "My_Classmates/content/My_Classmates_small"

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

def build_model():

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

    model = build_model()

    model(np.zeros((1,224,224,3)))

    model.load_weights(
        "face_encoder.weights.h5",
        by_name=True,
        skip_mismatch=True
    )

    return model

model = load_model()

# -------------------------
# preprocessing
# -------------------------

def preprocess_image(img):

    img = img.convert("RGB").resize((224,224))
    img = ImageOps.equalize(img)

    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0)

# -------------------------
# cosine distance
# -------------------------

def cosine_distance(a,b):
    return 1 - np.dot(a,b)

# -------------------------
# טעינת embeddings
# -------------------------

@st.cache_data
def load_reference_embeddings():

    embeddings = {}
    student_images = {}
    centroid_embeddings = {}

    for student in os.listdir(REFERENCE_DIR):

        student_path = os.path.join(REFERENCE_DIR, student)

        if os.path.isdir(student_path):

            embs = []
            first_img = None

            for file in os.listdir(student_path):

                if file.lower().endswith((".jpg",".png",".jpeg")):

                    path = os.path.join(student_path,file)

                    img = Image.open(path)
                    img = ImageOps.exif_transpose(img)

                    if first_img is None:
                        first_img = img

                    emb = model.predict(preprocess_image(img),verbose=0)[0]
                    emb = emb / np.linalg.norm(emb)

                    embs.append(emb)

            if embs:

                embeddings[student] = embs
                student_images[student] = first_img

                centroid = np.mean(embs,axis=0)
                centroid = centroid / np.linalg.norm(centroid)

                centroid_embeddings[student] = centroid

    return embeddings, centroid_embeddings, student_images


reference_embeddings, centroid_embeddings, student_images = load_reference_embeddings()

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

    H,W,_ = img.shape

    for det in detections:

        x,y,w,h = det["box"]

        pad = int(max(w,h)*0.35)

        x1 = max(0,x-pad)
        y1 = max(0,y-pad)

        x2 = min(W,x+w+pad)
        y2 = min(H,y+h+pad)

        face = img[y1:y2 , x1:x2]

        if face.size == 0:
            continue

        face = cv2.resize(face,(224,224))

        faces.append({
            "face":Image.fromarray(face),
            "box":(x1,y1,x2-x1,y2-y1)
        })

    return faces,img

# -------------------------
# זיהוי פנים עם KNN
# -------------------------

def recognize_face(emb,threshold=0.35,k=5):

    distances = []

    for name,ref_embs in reference_embeddings.items():

        for r in ref_embs:

            d = cosine_distance(emb,r)

            distances.append((name,d))

    distances.sort(key=lambda x:x[1])

    top = distances[:k]

    votes = {}

    for name,d in top:

        if d < threshold:
            votes[name] = votes.get(name,0)+1

    if not votes:
        return None

    return max(votes,key=votes.get)

# -------------------------
# העלאת תמונה
# -------------------------

st.subheader("העלי תמונת כיתה")

file = st.file_uploader("Upload class photo",type=["jpg","jpeg","png"])

# -------------------------
# זיהוי
# -------------------------

if st.button("בדוק נוכחות"):

    if file is None:
        st.warning("יש להעלות תמונה")
        st.stop()

    class_image = Image.open(file)
    class_image = ImageOps.exif_transpose(class_image)

    faces,original_img = extract_faces(class_image)

    present_students = {}
    recognized_faces = []

    for data in faces:

        img = data["face"]
        box = data["box"]

        emb = model.predict(preprocess_image(img),verbose=0)[0]
        emb = emb / np.linalg.norm(emb)

        name = recognize_face(emb)

        if name and name not in present_students:
            present_students[name] = img

        recognized_faces.append({
            "name":name,
            "box":box
        })

    img_draw = cv2.cvtColor(original_img.copy(),cv2.COLOR_RGB2BGR)

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

    st.image(cv2.cvtColor(img_draw,cv2.COLOR_BGR2RGB),use_column_width=True)

    missing_students = [s for s in STUDENT_ROSTER if s not in present_students]

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        st.header(f"✅ נוכחים ({len(present_students)})")

        cols = st.columns(3)

        for i,(name,img) in enumerate(present_students.items()):

            with cols[i%3]:
                st.write(name)
                st.image(img,width=120)

    with col2:

        st.header(f"❌ חסרים ({len(missing_students)})")

        cols = st.columns(3)

        for i,name in enumerate(missing_students):

            with cols[i%3]:
                st.write(name)

                if name in student_images:
                    st.image(student_images[name],width=120)
