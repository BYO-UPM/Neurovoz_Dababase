import pandas
import os
import io
import numpy as np


# Read the text transcriptions
path_hc = "./data/textHC.dat"
path_pd = "./data/textPD.dat"


# Get all audio files paths
path_audio = "./data/audios"

# read transcriptions (they have accents, so we need to use latin1 encoding)
with open(path_hc, "r", encoding="latin1") as f:
    text_hc = f.readlines()

# For each line, get the name of the file
filenames = [x.split(" ")[0] for x in text_hc]
# Reorder the filenames, they are currently NEHCXXXX_text., their corresponding audios are named HC_text_XXXX.wav
text_names = [
    x.split("0")[0][2:4] + "_" + x.split("_")[1] + "_" + x.split("C")[1][:4] + ".txt"
    for x in filenames
]

real_transcription = [x.split("-")[1][1:-2] for x in text_hc]

# Now, save in "data/transcriptions" the transcriptions
for i in range(len(text_names)):
    with open(os.path.join("data/transcriptions", text_names[i]), "w") as f:
        f.write(real_transcription[i])


# Now, the same but for PD
with open(path_pd, "r", encoding="latin1") as f:
    text_pd = f.readlines()

# For each line, get the name of the file
filenames = [x.split(" ")[0] for x in text_pd]
# Reorder the filenames, they are currently NEHCXXXX_text., their corresponding audios are named HC_text_XXXX.wav
text_names = [
    x.split("0")[0][2:4] + "_" + x.split("_")[1] + "_" + x.split("D")[1][:4] + ".txt"
    for x in filenames
]

real_transcription = [x.split("-")[1][1:-2] for x in text_pd]

# Now, save in "data/transcriptions" the transcriptions
for i in range(len(text_names)):
    with open(os.path.join("data/transcriptions", text_names[i]), "w") as f:
        f.write(real_transcription[i])
