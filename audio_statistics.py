import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Get total hours of audio registered
    import librosa

    total_hours_hc = 0
    for audio in audios_hc:
        y, sr = librosa.load(audio)
        total_hours_hc += librosa.get_duration(y=y, sr=sr)

    total_hours_pd = 0
    for audio in audios_pd:
        y, sr = librosa.load(audio)
        total_hours_pd += librosa.get_duration(y=y, sr=sr)

    print(f"Total hours of audio in healthy controls: {total_hours_hc / 3600}")
    print(f"Total hours of audio in patients: {total_hours_pd / 3600}")
    print(f"Total hours of audio: {(total_hours_hc + total_hours_pd) / 3600}")

    # The filename of the audio is composed by condition_audiotype_patientId.wav
    # We can split the filename by "_" and get the condition, audiotype and patientId

    # Select all vowels. They have an audiotype of len 2, which is the vowel and the recording number
    vowels_hc = [audio for audio in audios_hc if len(audio.split("_")[1]) == 2]
    vowels_pd = [audio for audio in audios_pd if len(audio.split("_")[1]) == 2]

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

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df_pd.groupby("ID").first()["UPDRS scale"], kde=True, color="skyblue")
    plt.title("Distribution of UPDRS Scale")

    # Plotting the distribution of H-Y stadium
    plt.subplot(1, 2, 2)
    sns.countplot(x="H-Y Stadium", data=df_pd.groupby("ID").first(), palette="Set2")
    plt.title("Distribution of H-Y Stadium")

    plt.tight_layout()
    plt.show()


def main():
    general_statistic()


if __name__ == "__main__":
    main()
