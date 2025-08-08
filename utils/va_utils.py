import json
import os

# Load AU landmark mapping
with open(os.path.join("mappings", "au_landmark_mapping.json"), "r") as f:
    AU_LANDMARK_MAPPING = json.load(f)

# Valence–Arousal mapping
# Values are illustrative; you can fine-tune per AU
AU_TO_VA = {
    "AU1": (-0.3, 0.4),   # Inner Brow Raiser
    "AU2": (0.2, 0.6),    # Outer Brow Raiser
    "AU4": (-0.6, 0.6),   # Brow Lowerer
    "AU5": (-0.4, 0.6),   # Upper Lid Raiser
    "AU6": (0.7, 0.4),    # Cheek Raiser
    "AU7": (-0.2, 0.3),   # Lid Tightener
    "AU9": (-0.6, 0.5),   # Nose Wrinkler
    "AU10": (0.3, 0.4),   # Upper Lip Raiser
    "AU11": (0.1, 0.2),   # Nasolabial Deepener
    "AU12": (0.6, 0.3),   # Lip Corner Puller (Smile)
    "AU13": (0.4, 0.3),   # Cheek Puffer
    "AU14": (0.2, 0.2),   # Dimpler
    "AU15": (-0.5, 0.2),  # Lip Corner Depressor
    "AU16": (-0.4, 0.1),  # Lower Lip Depressor
    "AU17": (-0.3, 0.2),  # Chin Raiser
    "AU18": (0.1, 0.3),   # Lip Pucker
    "AU20": (-0.7, 0.6),  # Lip Stretcher
    "AU22": (0.1, 0.3),   # Lip Funneler
    "AU23": (0.0, 0.2),   # Lip Tightener
    "AU24": (0.0, 0.1),   # Lip Pressor
    "AU25": (0.2, 0.4),   # Lips Part
    "AU26": (0.3, 0.7),   # Jaw Drop
    "AU27": (0.2, 0.8),   # Mouth Stretch
    "AU28": (0.0, 0.1),   # Lip Suck
    "AU43": (-0.1, -0.3)  # Eyes Closed
}

def estimate_valence_arousal(au_list):
    """Calculate average valence and arousal from a list of AUs."""
    if not au_list:
        return (0.0, 0.0)
    values = [AU_TO_VA[au] for au in au_list if au in AU_TO_VA]
    if not values:
        return (0.0, 0.0)
    valence, arousal = zip(*values)
    return sum(valence) / len(valence), sum(arousal) / len(arousal)
