AU_to_VA = {
    "AU6": (0.7, 0.4),  # High valence, medium arousal
    "AU12": (0.6, 0.3),
    "AU4": (-0.6, 0.6),
    "AU1": (-0.3, 0.4),
    "AU15": (-0.5, 0.2),
    "AU26": (0.3, 0.7),
    "AU9": (-0.6, 0.5),
    "AU2": (0.2, 0.6),
    "AU5": (-0.4, 0.6),
    "AU20": (-0.7, 0.6)
}

def estimate_valence_arousal(au_list):
    if not au_list: return (0.0, 0.0)
    valence, arousal = zip(*[AU_to_VA[au] for au in au_list if au in AU_to_VA])
    return sum(valence)/len(valence), sum(arousal)/len(arousal)
