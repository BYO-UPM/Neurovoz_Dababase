import pandas as pd
import numpy as np


def load_data():
    """
    Load the dataset from the CSV file.
    """
    df_hc = pd.read_csv("data/data_hc.csv")
    df_pd = pd.read_csv("data/data_pd.csv")
    return df_hc, df_pd


def general_statistic():
    """
    Print the general statistic of the dataset.
    """
    df_hc, df_pd = load_data()

    # Read all audios, the path to them is in the column "Audio"
    audios_hc = df_hc["Audio"].tolist()
    audios_pd = df_pd["Audio"].tolist()

    # The filename of the audio is composed by condition_audiotype_patientId.wav
    # We can split the filename by "_" and get the condition, audiotype and patientId

    # Select all vowels. They have an audiotype of len 2, which is the vowel and the recording number
    vowels_hc = [audio for audio in audios_hc if len(audio.split("_")[1]) == 2]
    vowels_pd = [audio for audio in audios_pd if len(audio.split("_")[1]) == 2]

    # Check the mean and std duration of the vowels using librosa
    import librosa

    # Use try and catch to avoid errors in the audios
    durations_hc = []
    failed_audios = []
    for audio in vowels_hc:
        try:
            y, sr = librosa.load(audio)
            durations_hc.append(librosa.get_duration(y=y, sr=sr))
        except:
            failed_audios.append(audio)
            pass
    print(f"Mean duration of vowels in healthy controls: {np.mean(durations_hc)}")
    print(f"Std duration of vowels in healthy controls: {np.std(durations_hc)}")
    print(f"Failed audios in healthy controls: {failed_audios}")

    durations_pd = []
    failed_audios = []
    for audio in vowels_pd:
        try:
            y, sr = librosa.load(audio)
            durations_pd.append(librosa.get_duration(y=y, sr=sr))
        except:
            failed_audios.append(audio)
            pass
    print(f"Mean duration of vowels in patients: {np.mean(durations_pd)}")
    print(f"Std duration of vowels in patients: {np.std(durations_pd)}")
    print(f"Failed audios in patients: {failed_audios}")

    # Select all monologues. Their audiotype is "ESPONTANEA".
    monologues_hc = [
        audio for audio in audios_hc if audio.split("_")[1] == "ESPONTANEA"
    ]
    monologues_pd = [
        audio for audio in audios_pd if audio.split("_")[1] == "ESPONTANEA"
    ]

    # Check the mean and std duration of the monologues using librosa
    durations_hc = []
    failed_audios = []
    for audio in monologues_hc:
        try:
            y, sr = librosa.load(audio)
            durations_hc.append(librosa.get_duration(y=y, sr=sr))
        except:
            failed_audios.append(audio)
            pass

    print(f"Mean duration of monologues in healthy controls: {np.mean(durations_hc)}")
    print(f"Std duration of monologues in healthy controls: {np.std(durations_hc)}")
    print(f"Failed audios in healthy controls: {failed_audios}")

    durations_pd = []
    failed_audios = []
    for audio in monologues_pd:
        try:
            y, sr = librosa.load(audio)
            durations_pd.append(librosa.get_duration(y=y, sr=sr))
        except:
            failed_audios.append(audio)
            pass
    print(f"Mean duration of monologues in patients: {np.mean(durations_pd)}")
    print(f"Std duration of monologues in patients: {np.std(durations_pd)}")
    print(f"Failed audios in patients: {failed_audios}")

    # Now select all the sentences. That is, all that has not been selected yet
    sentences_hc = [
        audio
        for audio in audios_hc
        if audio not in vowels_hc and audio not in monologues_hc
    ]
    sentences_pd = [
        audio
        for audio in audios_pd
        if audio not in vowels_pd and audio not in monologues_pd
    ]

    # Check the mean and std duration of the sentences using librosa
    durations_hc = []
    failed_audios = []
    for audio in sentences_hc:
        try:
            y, sr = librosa.load(audio)
            durations_hc.append(librosa.get_duration(y=y, sr=sr))
        except:
            failed_audios.append(audio)
            pass
    print(f"Mean duration of sentences in healthy controls: {np.mean(durations_hc)}")
    print(f"Std duration of sentences in healthy controls: {np.std(durations_hc)}")
    print(f"Failed audios in healthy controls: {failed_audios}")

    durations_pd = []
    failed_audios = []
    for audio in sentences_pd:
        try:
            y, sr = librosa.load(audio)
            durations_pd.append(librosa.get_duration(y=y, sr=sr))
        except:
            failed_audios.append(audio)
            pass
    print(f"Mean duration of sentences in patients: {np.mean(durations_pd)}")
    print(f"Std duration of sentences in patients: {np.std(durations_pd)}")
    print(f"Failed audios in patients: {failed_audios}")


def main():
    general_statistic()
