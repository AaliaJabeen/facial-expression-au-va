import cv2
import mediapipe as mp
from utils.face_cropper import crop_face
from utils.va_utils import estimate_valence_arousal
from models.cnn_model import EmotionCNN
from torchvision import transforms
from PIL import Image
import torch
import json

# Load model
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("models/emotion_cnn.pth", map_location=device))
model.eval()

# Load mapping
with open("mappings/emotion_to_au.json") as f:
    emotion_to_au = json.load(f)

# Heisenberg.jpg 
# jacky chan.jpg
# kevin hart.jpg
# khal drogo.jpg
# rock.jpg
# Load image
image_path = "data/rock.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Crop face for emotion prediction
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

# Get AU and VA
aus = emotion_to_au[str(pred_idx)]
valence, arousal = estimate_valence_arousal(aus)

# Draw landmarks with MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

results = face_mesh.process(image_rgb)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            x = int(lm.x * image_bgr.shape[1])
            y = int(lm.y * image_bgr.shape[0])
            cv2.circle(image_bgr, (x, y), 1, (0, 255, 0), -1)

# Add text
cv2.putText(image_bgr, f'Emotion: {predicted_emotion}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(image_bgr, f'AUs: {", ".join(aus)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
cv2.putText(image_bgr, f'Valence: {valence:.2f}  Arousal: {arousal:.2f}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

# Show image
cv2.imshow("Emotion + VA + Landmarks", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
