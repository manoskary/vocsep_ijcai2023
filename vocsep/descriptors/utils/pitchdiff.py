import numpy as np
from partitura.musicanalysis import estimate_voices


def get_pitchdiff(part, note_array):
    features = ["estimated_voices", "pitch_diff_left", "pitch_diff_right"]
    voices = estimate_voices(note_array)
    note_array["voice"] = voices
    pitchdiff = np.zeros(len(note_array))

    for voice in np.unique(voices):
        idx = np.where(note_array["voice"] == voice)
        notes = note_array[idx]["pitch"]
        pitchdiff[idx] = np.r_[notes[1:], notes[-1]] - notes
    pitchdiff = np.expand_dims(pitchdiff, axis=1)
    out = np.hstack((np.expand_dims(voices, axis=1), pitchdiff, -pitchdiff))
    return out, features
