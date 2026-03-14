import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from mtcnn import MTCNN
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# -------------------------
# הגדרות
# -------------------------

st.set_page_config(layout="wide")
st.title("Face Crop + Embedding Debug")

DEBUG_DIR = "debug_faces"
os.makedirs(DEBUG_DIR, exist_ok=True)

REFERENCE_DIR = "My_Classmates/content/My_Classmates_small"

detector = MTCNN()

# -------------------------
# מודל embedding
# -------------------------

class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


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
        layers.Dense(128),
        L2Normalize()
    ])

    return model


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

def preprocess(img):

    img = img.resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0

    return np.expand_dims(arr,0)


# -------------------------
# cosine distance
# -------------------------

def cosine_distance(a,b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return 1 - np.dot(a,b)


# -------------------------
# טעינת מאגר embeddings
# -------------------------

@st.cache_data
def load_reference_embeddings():

    embeddings = {}

    for student in os.listdir(REFERENCE_DIR):

        student_path = os.path.join(REFERENCE_DIR,student)

        if os.path.isdir(student_path):

            embs = []

            for file in os.listdir(student_path):

                if file.lower().endswith(("jpg","png","jpeg")):

                    path = os.path.join(student_path,file)

                    img = Image.open(path)
                    img = ImageOps.exif_transpose(img)

                    emb = model.predict(preprocess(img),verbose=0)[0]

                    embs.append(emb)

            if embs:
                embeddings[student] = embs

    return embeddings


reference_embeddings = load_reference_embeddings()

# -------------------------
# יישור פנים
# -------------------------

def align_face(face,left_eye,right_eye):

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy,dx))

    center = tuple(np.array(face.shape[1::-1]) / 2)

    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)

    aligned = cv2.warpAffine(
        face,
        rot_mat,
        face.shape[1::-1],
        flags=cv2.INTER_CUBIC
    )

    return aligned


# -------------------------
# חיתוך פנים
# -------------------------

def extract_faces(image):

    image = image.convert("RGB")
    img = np.array(image)

    detections = detector.detect_faces(img)

    faces = []

    H,W,_ = img.shape

    for i,det in enumerate(detections):

        x,y,w,h = det["box"]

        x = max(0,x)
        y = max(0,y)

        pad = int(max(w,h)*0.35)

        x1 = max(0,x-pad)
        y1 = max(0,y-pad)

        x2 = min(W,x+w+pad)
        y2 = min(H,y+h+pad)

        face = img[y1:y2 , x1:x2]

        if face.size == 0:
            continue

        size_before = face.shape[:2]

        keypoints = det["keypoints"]

        left_eye = (keypoints["left_eye"][0]-x1,
                    keypoints["left_eye"][1]-y1)

        right_eye = (keypoints["right_eye"][0]-x1,
                     keypoints["right_eye"][1]-y1)

        face = align_face(face,left_eye,right_eye)

        face_resized = cv2.resize(face,(224,224))

        face_pil = Image.fromarray(face_resized)

        save_path = os.path.join(DEBUG_DIR,f"face_{i}.jpg")

        face_pil.save(save_path)

        faces.append({
            "original":face,
            "resized":face_resized,
            "size":size_before,
            "pil":face_pil,
            "path":save_path,
            "box":(x1,y1,x2-x1,y2-y1)
        })

    return faces,img


# -------------------------
# העלאת תמונה
# -------------------------

uploaded = st.file_uploader(
    "Upload class photo",
    type=["jpg","jpeg","png"]
)

if uploaded:

    class_image = Image.open(uploaded)
    class_image = ImageOps.exif_transpose(class_image)

    faces,img = extract_faces(class_image)

    st.write(f"זוהו {len(faces)} פנים")

    draw = img.copy()

    for f in faces:

        x,y,w,h = f["box"]

        cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)

    st.image(draw)

    st.divider()

    cols = st.columns(3)

    for i,f in enumerate(faces):

        with cols[i % 3]:

            st.write(f"Face {i}")

            st.write("גודל לפני resize:")
            st.write(f["size"])

            st.image(f["original"],caption="לפני resize")

            st.image(f["resized"],caption="אחרי resize")

            st.write("קובץ לבדיקה:")
            st.code(f["path"])

            # -------------------------
            # בדיקת embedding
            # -------------------------

            emb = model.predict(
                preprocess(f["pil"]),
                verbose=0
            )[0]

            best_name = None
            best_dist = 999

            for name,embs in reference_embeddings.items():

                for ref in embs:

                    d = cosine_distance(emb,ref)

                    if d < best_dist:
                        best_dist = d
                        best_name = name

            st.write("זיהוי קרוב ביותר:")

            st.write(best_name)

            st.write("distance:")

            st.write(best_dist)
