import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image, ImageOps
import numpy as np
import os
import zipfile
import cv2

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


def build_model():

    base_model = MobileNetV2(
        input_shape=(128,128,3),
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


@st.cache_resource
def load_model():

    model = build_model()

    model(np.zeros((1,128,128,3)))

    model.load_weights(
        "face_encoder.weights.h5",
        by_name=True,
        skip_mismatch=True
    )

    return model


model = load_model()
st.success("המודל נטען בהצלחה")


def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((128,128))

    arr = np.array(img).astype(np.float32) / 255.0

    return np.expand_dims(arr, axis=0)


def cosine_distance(a,b):

    return 1 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


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

                    student_embeddings.append(emb)

            if student_embeddings:
                embeddings[student] = np.mean(student_embeddings, axis=0)

    return embeddings


reference_embeddings = load_reference_embeddings()

st.info(f"נמצאו {len(reference_embeddings)} תלמידים במאגר")


@st.cache_resource
def load_face_detector():

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    return detector


face_detector = load_face_detector()


def extract_faces(image):

    image = image.convert("RGB")
    img = np.array(image)

    max_size = 900
    h, w = img.shape[:2]

    if max(h, w) > max_size:

        scale = max_size / max(h, w)

        img = cv2.resize(
            img,
            (int(w*scale), int(h*scale))
        )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detections = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=6,
        minSize=(60,60)
    )

    faces = []

    for (x,y,w,h) in detections:

        pad = int(w * 0.25)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)

        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        face = img[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face = cv2.resize(face,(128,128))

        face_img = Image.fromarray(face)

        faces.append({
            "face": face_img,
            "box": (x,y,w,h)
        })

    return faces, img


with st.sidebar:

    st.header("הגדרות")

    threshold = st.slider(
        "Similarity Threshold",
        0.2,
        0.9,
        0.55
    )

    st.write("תלמידים בכיתה")

    for s in STUDENT_ROSTER:
        st.write(s)


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

        best_name = None
        best_dist = 1.0

        for name, ref_emb in reference_embeddings.items():

            dist = cosine_distance(emb, ref_emb)

            if dist < best_dist:

                best_dist = dist
                best_name = name

        if best_dist < threshold:

            if best_name not in present_students:
                present_students[best_name] = img

        recognized_faces.append({
            "name": best_name if best_dist < threshold else None,
            "box": box
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
                st.image(img,width=100)

    with col2:

        st.header(f"❌ חסרים ({len(missing_students)})")

        if missing_students:

            for s in missing_students:
                st.write(s)

        else:
            st.success("כולם נוכחים")
