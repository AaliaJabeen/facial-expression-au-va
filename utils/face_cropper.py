import cv2
from PIL import Image

def crop_face(image_path, cascade_path="utils/haarcascade_frontalface_default.xml"):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"❌ Image not found at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No face detected, returning full grayscale image.")
        return Image.fromarray(gray)

    # Take the largest face
    x, y, w, h = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
    face_img = gray[y:y+h, x:x+w]
    return Image.fromarray(face_img)
