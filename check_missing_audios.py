import os
import pandas as pd
from collections import defaultdict
import missingno as msno
import matplotlib.pyplot as plt


# Function to get the list of all .wav files in the directory
def get_wav_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".wav")]


# Function to extract patient ID and audio material from the filename
def parse_filename(filename):
    parts = filename.split("_")
    condition = parts[0]
    audio_material = parts[1]
    patient_id = parts[-1].split(".")[0]
    return condition, audio_material, patient_id


# Function to create the DataFrame
def create_audio_dataframe(directory):
    wav_files = get_wav_files(directory)
    patient_files = defaultdict(dict)
    patient_conditions = {}

    for wav_file in wav_files:
        condition, audio_material, patient_id = parse_filename(wav_file)
        patient_files[patient_id][audio_material] = wav_file
        patient_conditions[patient_id] = condition

    print("Total number of audios:", len(wav_files))

    # Get the list of all unique audio materials
    all_audio_materials = set()
    for files in patient_files.values():
        all_audio_materials.update(files.keys())

    all_audio_materials = sorted(all_audio_materials)

    # Create the DataFrame
    df = pd.DataFrame(columns=["Patient_ID", "Condition"] + all_audio_materials)
    for patient_id, materials in patient_files.items():
        row = {"Patient_ID": patient_id, "Condition": patient_conditions[patient_id]}
        for material in all_audio_materials:
            row[material] = materials.get(material, float("NaN"))
        # Use pd concat
        df = pd.concat(
            [
                df,
                pd.DataFrame(row, index=[0]),
            ],
            ignore_index=True,
        )

    # Check that all wav_file are in the df
    audio_not_found = set(wav_files) - set(df.iloc[:, 1:].values.flatten())

    return df


# Function to plot missing data
def plot_missing_data(df):
    plt.figure(figsize=(12, 8))
    msno.bar(df.set_index("Patient_ID"))
    plt.savefig("missing_audio.png", dpi=300)
    plt.show()


# Main function to specify the directory and create the DataFrame
def main():
    directory = "/NeuroVoz/data/data/audios"  # Change this to your directory path
    df = create_audio_dataframe(directory)
    df.to_csv("audio_materials_per_patient.csv", index=False)
    print(df)
    plot_missing_data(df)

    # Make PatientID the index
    df = df.set_index("Patient_ID")


if __name__ == "__main__":
    main()
