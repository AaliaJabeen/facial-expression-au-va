from models.cnn_model import EmotionCNN
from utils.va_utils import estimate_valence_arousal
import torchvision.transforms as transforms
from PIL import Image
import json
import torch

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("models/emotion_cnn.pth"))
model.eval()

# Load image
img = Image.open("data/ishrat.bmp").convert("L")
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    predicted = torch.argmax(output, dim=1).item()

# Map emotion index → AU
with open("mappings/emotion_to_au.json") as f:
    emotion_to_au = json.load(f)

aus = emotion_to_au[str(predicted)]
valence, arousal = estimate_valence_arousal(aus)

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print(f"Predicted Emotion: {class_names[predicted]}")
print(f"AUs: {aus}")
print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")
