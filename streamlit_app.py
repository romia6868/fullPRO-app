import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import os
import zipfile
from retinaface import RetinaFace

st.set_page_config(page_title="מערכת נוכחות חכמה", layout="wide")
st.title("📸 מערכת נוכחות חכמה")

ZIP_PATH = "My_Classmates_small.zip"
EXTRACT_PATH = "My_Classmates"

if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH,'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

REFERENCE_DIR = "My_Classmates/content/My_Classmates_small"

STUDENT_ROSTER = ['Maayan','Tomer','Roei','Zohar','Ilay']

class L2Normalize(tf.keras.layers.Layer):
    def call(self,inputs):
        return tf.math.l2_normalize(inputs,axis=1)

def build_embedding_model():

    base_model = MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512,activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256,activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(128),
        L2Normalize()
    ])

    return model

@st.cache_resource
def load_model():

    model = build_embedding_model()

    model(np.zeros((1,224,224,3)))

    model.load_weights(
        "face_encoder.weights.h5",
        by_name=True,
        skip_mismatch=True
    )

    return model

model = load_model()

st.success("המודל נטען")

def preprocess_image(img):

    img = img.convert("RGB").resize((224,224))

    arr = np.array(img).astype(np.float32)

    arr = preprocess_input(arr)

    return np.expand_dims(arr,axis=0)

def cosine_similarity(a,b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return np.dot(a,b)

MAX_SIZE = 1200

def resize_for_detection(image):

    w,h = image.size

    if max(w,h) > MAX_SIZE:

        scale = MAX_SIZE / max(w,h)

        new_w = int(w*scale)
        new_h = int(h*scale)

        image = image.resize((new_w,new_h))

    return image

def extract_faces(image):

    image = resize_for_detection(image)

    image = image.convert("RGB")
    img = np.array(image)

    detections = RetinaFace.detect_faces(img, threshold=0.6)

    faces = []

    if isinstance(detections,dict):

        for key in detections:

            identity = detections[key]

            x1,y1,x2,y2 = identity["facial_area"]

            w = x2-x1

            if w < 35:
                continue

            pad = int(0.25*w)

            x1 = max(0,x1-pad)
            y1 = max(0,y1-pad)
            x2 = min(img.shape[1],x2+pad)
            y2 = min(img.shape[0],y2+pad)

            face = img[y1:y2 , x1:x2]

            if face.size == 0:
                continue

            face_img = Image.fromarray(face).resize((224,224))

            faces.append({
                "face":face_img,
                "box":(x1,y1,x2-x1,y2-y1)
            })

    return faces,image

@st.cache_data
def load_reference_embeddings():

    embeddings = {}

    for student in os.listdir(REFERENCE_DIR):

        student_path = os.path.join(REFERENCE_DIR,student)

        if os.path.isdir(student_path):

            student_embs = []

            for file in os.listdir(student_path):

                if file.lower().endswith((".jpg",".jpeg",".png")):

                    img = Image.open(os.path.join(student_path,file))
                    img = ImageOps.exif_transpose(img)

                    faces,_ = extract_faces(img)

                    for f in faces:

                        emb = model.predict(
                            preprocess_image(f["face"]),
                            verbose=0
                        )[0]

                        emb = emb / np.linalg.norm(emb)

                        student_embs.append(emb)

            if student_embs:
                embeddings[student] = student_embs

    return embeddings

reference_embeddings = load_reference_embeddings()

st.info(f"נמצאו {len(reference_embeddings)} תלמידים במאגר")

with st.sidebar:

    st.header("הגדרות")

    threshold = st.slider(
        "Similarity Threshold",
        0.7,
        1.0,
        0.85
    )

st.subheader("העלי תמונת כיתה")

class_file = st.file_uploader(
    "Upload class photo",
    type=["jpg","jpeg","png"]
)

if st.button("בדוק נוכחות"):

    if class_file is None:
        st.warning("יש להעלות תמונה")
        st.stop()

    class_image = Image.open(class_file)
    class_image = ImageOps.exif_transpose(class_image)

    faces,original_img = extract_faces(class_image)

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

        best_name = None
        best_score = -1

        for name,ref_list in reference_embeddings.items():

            for ref_emb in ref_list:

                score = cosine_similarity(emb,ref_emb)

                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < threshold:
            best_name = None

        if best_name and best_name not in present_students:
            present_students[best_name] = img

        recognized_faces.append({
            "name":best_name,
            "box":box,
            "score":best_score
        })

    img_draw = original_img.copy()
    draw = ImageDraw.Draw(img_draw)

    for face in recognized_faces:

        x,y,w,h = face["box"]

        name = face["name"] if face["name"] else "Unknown"
        score = round(face["score"],2)

        label = f"{name} {score}"

        draw.rectangle([x,y,x+w,y+h], outline="green", width=3)
        draw.text((x,y-15), label, fill="green")

    st.subheader("תוצאת זיהוי")

    st.image(img_draw,use_column_width=True)

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
