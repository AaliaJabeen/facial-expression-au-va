---
# Facial Expression to Valence-Arousal Mapping

A PyTorch-based project that detects facial expressions from the FER2013 dataset using a CNN model, maps them to Facial Action Units (AUs), and estimates Valence-Arousal (VA) emotional dimensions.

---

## 🧠 Features

- Train a CNN on FER2013 emotion images
- Automatically map predicted emotions to Facial Action Units (AUs)
- Estimate Valence and Arousal levels from AUs
- Clean, modular code with inference support

---

## 🗂️ Project Structure

```

facial-emotion-va/
├── data/                         # FER2013 dataset (images in subfolders)
│   ├── train/angry, happy...
│   └── test/
├── models/
│   ├── emotion_cnn.pth           # Saved model weights
│   └── cnn_model.py              # CNN model architecture
├── utils/
│   ├── preprocess.py             # Data loading/transforms
│   └── va_utils.py               # AU → VA mapping logic
├── mappings/
│   ├── emotion_to_au.json
├── notebooks/
│   └── EDA_and_Training.ipynb
├── main.py                       # Inference script
├── check_model.py                # Model tester
└── README.md

````

---

## 📦 Setup Instructions

1. **Clone the repo & create environment**

```bash
git clone https://github.com/iammasoodalam/facial-expression-au-va.git
cd facial-expression-au-va
python -m venv .venv
Scripts\activate        # or (in linux) source bin/activate
pip install -r requirements.txt
````

2. **Prepare dataset**

Download the FER2013 image version from:
[https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Extract it as:

```
data/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    └── ...
```
or

```bash
python download_dataset.py
```

3. **Train the model**

Run the scripts in the notebook one-by-one

4. **Run inference on a sample image**

Save your image in the data folder, and put the name of the image in check_model.py.

```bash
python check_model.py
```

---

## 📊 Valence-Arousal Mapping

We use predefined rules to estimate VA scores from Action Units (AUs) based on research literature.

* `emotion → AUs` → defined in `mappings/emotion_to_au.json`
* `AUs → (Valence, Arousal)` → defined in `mappings/au_to_va.json`

---

## 💡 Example Output

```bash
🔍 Inference Result
------------------------
Image        : sample_face.jpg
Emotion      : happy
Action Units : ['AU6', 'AU12']
Valence      : 0.65
Arousal      : 0.42
```

---

## 🔧 Requirements

* Python 3.8+
* PyTorch
* torchvision
* matplotlib
* opencv-python (optional)
* Jupyter (optional for training)

Install everything using:

```bash
pip install -r requirements.txt
```

---

## 📌 Future Improvements

* Add face detection + auto-cropping
* Real-time webcam inference with OpenCV
* Streamlit app for web demo

---

## 📑 License

MIT License. Free to use and modify.