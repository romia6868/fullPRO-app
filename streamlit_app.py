import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from mtcnn import MTCNN
import os

# -------------------------
# הגדרות
# -------------------------

st.set_page_config(page_title="Face Crop Debug", layout="wide")
st.title("בדיקת חיתוך פנים מתמונה כיתתית")

detector = MTCNN()

DEBUG_DIR = "debug_faces"
os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------------
# יישור פנים לפי עיניים
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

        original_size = face.shape[:2]

        keypoints = det["keypoints"]

        left_eye = keypoints["left_eye"]
        right_eye = keypoints["right_eye"]

        left_eye = (left_eye[0]-x1,left_eye[1]-y1)
        right_eye = (right_eye[0]-x1,right_eye[1]-y1)

        face = align_face(face,left_eye,right_eye)

        # resize
        face_resized = cv2.resize(face,(224,224))

        # חשוב מאוד: המרת צבעים
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        face_original_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face_pil = Image.fromarray(face_resized)

        save_path = os.path.join(DEBUG_DIR,f"face_{i}.jpg")
        face_pil.save(save_path)

        faces.append({
            "face_original":face_original_rgb,
            "face_resized":face_resized,
            "size":original_size,
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

# -------------------------
# עיבוד
# -------------------------

if uploaded:

    class_image = Image.open(uploaded)
    class_image = ImageOps.exif_transpose(class_image)

    faces,original_img = extract_faces(class_image)

    st.write(f"זוהו {len(faces)} פנים")

    # ציור קופסאות
    draw = cv2.cvtColor(original_img.copy(),cv2.COLOR_RGB2BGR)

    for f in faces:

        x,y,w,h = f["box"]

        cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)

    st.subheader("זיהוי פנים בתמונה")

    st.image(
        cv2.cvtColor(draw,cv2.COLOR_BGR2RGB),
        use_column_width=True
    )

    st.divider()

    st.subheader("בדיקת הפנים שנחתכו")

    cols = st.columns(3)

    for i,f in enumerate(faces):

        with cols[i % 3]:

            st.write(f"Face {i}")

            st.write("גודל לפני resize:")
            st.write(f["size"])

            st.image(
                f["face_original"],
                caption="לפני resize"
            )

            st.image(
                f["face_resized"],
                caption="אחרי resize (224x224)"
            )

            st.write("קובץ לבדיקה באפליקציה השנייה:")

            st.code(f["path"])
