import cv2
import mediapipe as mp
from utils.face_cropper import crop_face
from utils.va_utils import estimate_valence_arousal
from models.cnn_model import EmotionCNN
from torchvision import transforms
from PIL import Image
import torch
import json
import os

# Load emotion classification model
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("models/emotion_cnn.pth", map_location=device))
model.eval()

# Load AU mappingskevikevin hartn hart
with open("mappings/emotion_to_au.json") as f:
    emotion_to_au = json.load(f)

with open("mappings/au_landmark_mapping.json") as f:
    au_landmark_mapping = json.load(f)

# Load input image
image_path = "data/jacky chan.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Preprocess image for emotion model
face_gray_pil = crop_face(image_path)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])
input_tensor = transform(face_gray_pil).unsqueeze(0).to(device)

# Predict emotion
with torch.no_grad():
    output = model(input_tensor)
    pred_idx = torch.argmax(output, dim=1).item()
    predicted_emotion = class_names[pred_idx]

# Get associated AUs and compute VA
aus = emotion_to_au[str(pred_idx)]
valence, arousal = estimate_valence_arousal(aus)

# AU visualization with landmark highlighting
def draw_au_regions(image, face_landmarks, aus, au_mapping):
    for au in aus:
        if au in au_mapping:
            landmark_indices = au_mapping[au]["landmarks"]
            description = au_mapping[au]["description"]
            
            landmark_labels = []  # For human-readable location text

            for idx in landmark_indices:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x = int(lm.x * image.shape[1])
                    y = int(lm.y * image.shape[0])
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                    landmark_labels.append(f'{idx}')

            # Prepare annotated label
            if landmark_indices:
                first_idx = landmark_indices[0]
                first_lm = face_landmarks.landmark[first_idx]
                x = int(first_lm.x * image.shape[1])
                y = int(first_lm.y * image.shape[0])
                label = f'{au}: {description} [{", ".join(landmark_labels)}]'
                cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Run MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
results = face_mesh.process(image_rgb)

# Draw base landmarks and AU regions
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Draw all landmarks in green
        for lm in face_landmarks.landmark:
            x = int(lm.x * image_bgr.shape[1])
            y = int(lm.y * image_bgr.shape[0])
            cv2.circle(image_bgr, (x, y), 1, (0, 255, 0), -1)
        # Draw AU-specific points
        draw_au_regions(image_bgr, face_landmarks, aus, au_landmark_mapping)

# Overlay emotion, AU, VA text
cv2.putText(image_bgr, f'Emotion: {predicted_emotion}', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(image_bgr, f'AUs: {", ".join(aus)}', (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(image_bgr, f'Valence: {valence:.2f}  Arousal: {arousal:.2f}', (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Show the result
cv2.imshow("Emotion + VA + AU Landmarks", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()