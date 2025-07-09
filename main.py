import torch
from models.cnn_model import EmotionCNN
from utils.preprocess import load_fer2013
from utils.va_utils import estimate_valence_arousal
import json

# Load model
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_cnn.pth"))
model.eval()

# Load sample
faces, labels = load_fer2013()
sample = torch.tensor(faces[0]).unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 48, 48)

# Predict emotion
with torch.no_grad():
    output = model(sample)
    predicted = torch.argmax(output, dim=1).item()

# Load AU mapping
with open("mappings/emotion_to_au.json") as f:
    emo_to_au = json.load()
aus = emo_to_au[str(predicted)]

# Estimate VA
val, aro = estimate_valence_arousal(aus)
print(f"Predicted Emotion: {predicted}, AUs: {aus}, VA: ({val:.2f}, {aro:.2f})")
