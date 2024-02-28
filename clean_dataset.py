import os
import glob
import numpy as np
import pandas as pd
from easynmt import EasyNMT


def find_wav_files(root_dir):
    """
    Find all .wav files in the specified directory and its subdirectories.
    """
    return glob.glob(os.path.join(root_dir, "*.wav"), recursive=True)


def extract_patient_ids(filenames):
    """
    Extract patient IDs (last 4 digits before the extension) from filenames.
    """
    patient_ids = [int(os.path.basename(filename)[-8:-4]) for filename in filenames]
    return patient_ids


def extract_condition(filenames):
    """
    Extract patients condition (PD or HC). They are the two first letters of the filename.
    """
    condition = [os.path.basename(filename)[:2] for filename in filenames]
    return condition


def calculate_general_statistics(patient_ids):
    """
    Calculate mean and standard deviation of audio files per patient.
    """
    unique_ids, counts = np.unique(patient_ids, return_counts=True)
    mean_count = np.mean(counts)
    std_dev = np.std(counts)
    return mean_count, std_dev, len(unique_ids), len(patient_ids)


def categorize_column(df, column):
    """
    Categorize a column in a dataframe to anonymize the data.
    """

    # The column is a string, so first, make sure they are strings
    df[column] = df[column].astype(str)
    # Remove accents
    df[column] = (
        df[column]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.replace(" ", "")
        .str.lower()
        .str.replace(" ", "-")
        .str.replace("de", "")
        .str.replace("--", "-")
        .str.rstrip("-")
    )
    if (
        column == "Cephalic tremor"
        or column == "Mandibular tremor"
        or column == "Sialorrhoea"
        or column == "Dysphagia"
        or column == "Hypophonic voice"
    ):
        # if starts with si, we will fill it with "si"
        idx_si = df[column].str.lower().str.startswith("si")
        df.loc[idx_si, column] = "si"
        idx_si = df[column].str.lower().str.startswith("poco")
        df.loc[idx_si, column] = "si"

    # Ensure that nan is np.nan
    df[column] = df[column].infer_objects(copy=False).replace("nan", np.nan)

    if column == "Doctor":
        # Velasquez and Velazquez are the same
        df[column] = df[column].replace("Velasquez", "Velazquez")
    df[column] = df[column].astype("category")
    df[column] = df[column].cat.codes

    return df


def clean_excel(excel_path):
    data_pd = pd.read_excel(excel_path, sheet_name="PD")
    data_hc = pd.read_excel(excel_path, sheet_name="CONTROL")
    data_pd = data_pd.dropna(subset=["PAC"])
    data_hc = data_hc.dropna(subset=["PAC"])

    # Rename PAC to ID
    data_pd = data_pd.rename(columns={"PAC": "ID"})
    data_hc = data_hc.rename(columns={"PAC": "ID"})

    # Rename "Médico" to "Doctor"
    data_pd = data_pd.rename(columns={"Médico": "Doctor"})
    data_hc = data_hc.rename(columns={"Médico": "Doctor"})

    # Fill na with np.nan
    data_pd = data_pd.infer_objects(copy=False).fillna(np.nan)
    data_hc = data_hc.infer_objects(copy=False).fillna(np.nan)

    # Categorize Doctor to anonymize
    data_pd = categorize_column(data_pd, "Doctor")
    data_hc = categorize_column(data_hc, "Doctor")
    # Categorize Sexo
    data_pd = categorize_column(data_pd, "Sexo")
    data_hc = categorize_column(data_hc, "Sexo")

    # Substitute -1 for NaN
    data_pd = data_pd.replace(-1, np.nan)
    data_hc = data_hc.replace(-1, np.nan)

    # DROP NHC column for anonymization
    data_pd = data_pd.drop(columns=["NHC"])
    data_hc = data_hc.drop(columns=["NHC"])
    data_hc = data_hc.drop(columns=["Unnamed: 24"])

    # Translaste all dataframe to english
    model = EasyNMT("opus-mt")
    # Translate columns
    data_pd.columns = model.translate(
        data_pd.columns, source_lang="es", target_lang="en"
    )
    data_hc.columns = model.translate(
        data_hc.columns, source_lang="es", target_lang="en"
    )

    # Correct Age column: None should be np.nan and ? should be np.nan
    data_pd["Age"] = data_pd["Age"].replace("None", np.nan)
    data_pd["Age"] = data_pd["Age"].replace("?", np.nan)
    data_pd["Age"] = data_pd["Age"].astype(float)
    data_hc["Age"] = data_hc["Age"].replace("None", np.nan)
    data_hc["Age"] = data_hc["Age"].replace("?", np.nan)
    data_hc["Age"] = data_hc["Age"].astype(float)

    return data_pd, data_hc


def main():
    root_dir = "data/audios"  # Change this to your directory path
    wav_files = find_wav_files(root_dir)
    patient_ids = extract_patient_ids(wav_files)
    conditions = extract_condition(wav_files)
    mean_count, std_dev, num_patients, total_files = calculate_general_statistics(
        patient_ids
    )
    # Aggrupate patients by conditions
    pd_ids = [patient_ids[i] for i in range(len(patient_ids)) if conditions[i] == "PD"]
    hc_ids = [patient_ids[i] for i in range(len(patient_ids)) if conditions[i] == "HC"]
    unique_pd_ids, counts_pd = np.unique(pd_ids, return_counts=True)
    unique_hc_ids, counts_hc = np.unique(hc_ids, return_counts=True)

    excel_path = "data/raw/Datos pacientes_notas_total.xlsx"
    data_pd, data_hc = clean_excel(excel_path)
    unique_excel_pd_ids = data_pd["ID"].unique()
    unique_excel_hc_ids = data_hc["ID"].unique()

    # Matching patients: i.e. we have their audio and their data
    matching_pd = list(set(unique_pd_ids).intersection(unique_excel_pd_ids))
    matching_hc = list(set(unique_hc_ids).intersection(unique_excel_hc_ids))

    # Excluded patients: i.e. we have their audio but not their data
    unmatching_pd = list(set(unique_pd_ids).difference(unique_excel_pd_ids))
    unmatching_hc = list(set(unique_hc_ids).difference(unique_excel_hc_ids))

    # Excluded patients: i.e. we have their data but not their audio
    excluded_pd = list(set(unique_excel_pd_ids).difference(unique_pd_ids))
    excluded_hc = list(set(unique_excel_hc_ids).difference(unique_hc_ids))

    print(f"Matching PD: {len(matching_pd)}")
    print(f"Matching HC: {len(matching_hc)}")
    print(f"Unmatching PD: {len(unmatching_pd)}")
    print(f"Unmatching HC: {len(unmatching_hc)}")
    print(f"Excluded PD: {len(excluded_pd)}")
    print(f"Excluded HC: {len(excluded_hc)}")

    # Remove excluded and unmatching patients from the dataframes
    data_pd = data_pd[data_pd["ID"].isin(matching_pd)]
    data_hc = data_hc[data_hc["ID"].isin(matching_hc)]
    # Remove audios from unmatching patients
    patients_idx = np.where(np.isin(patient_ids, unmatching_pd))[0]
    wav_files = [wav_files[i] for i in range(len(wav_files)) if i not in patients_idx]
    patient_ids = [
        patient_ids[i] for i in range(len(patient_ids)) if i not in patients_idx
    ]
    conditions = [
        conditions[i] for i in range(len(conditions)) if i not in patients_idx
    ]
    # do the same for hc
    patients_idx = np.where(np.isin(patient_ids, unmatching_hc))[0]
    wav_files = [wav_files[i] for i in range(len(wav_files)) if i not in patients_idx]
    patient_ids = [
        patient_ids[i] for i in range(len(patient_ids)) if i not in patients_idx
    ]
    conditions = [
        conditions[i] for i in range(len(conditions)) if i not in patients_idx
    ]

    # Add a column to the dataframe with the path to the audio file which is:
    filename = [os.path.basename(file) for file in wav_files]
    # Add an audio column to the dataframe
    # data/clean + filename
    audio_path = [os.path.join("data/audios", file) for file in filename]
    # Select the hc audio pathes
    get_id_of_audio = lambda x: int(x[-8:-4])
    audio_ids_path_df = pd.DataFrame(
        {"ID": [get_id_of_audio(file) for file in filename], "Audio": audio_path}
    )
    # Merge the dataframess
    # Assure that data_pd ID is an integer
    data_pd["ID"] = data_pd["ID"].astype(int)
    data_pd = data_pd.merge(audio_ids_path_df, on="ID")
    # Assure that data_hc ID is an integer
    data_hc["ID"] = data_hc["ID"].astype(int)
    data_hc = data_hc.merge(audio_ids_path_df, on="ID")

    # Translate all str columsn to english
    model = EasyNMT("opus-mt")
    # Translate columns
    for column in ["Observations"]:

        data_pd[column] = model.translate(
            data_pd[column].astype(str), source_lang="es", target_lang="en"
        )

    for column in ["Observations"]:
        data_hc[column] = model.translate(
            data_hc[column].astype(str), source_lang="es", target_lang="en"
        )

    # Categorize Vocal tremor,Cephalic tremor,Mandibular tremor,Sialorrhoea,Dysphagia,Hypophonic voice columns
    data_pd = categorize_column(data_pd, "Vocal tremor")
    data_pd = categorize_column(data_pd, "Cephalic tremor")
    data_pd = categorize_column(data_pd, "Mandibular tremor")
    data_pd = categorize_column(data_pd, "Sialorrhoea")
    data_pd = categorize_column(data_pd, "Dysphagia")
    data_pd = categorize_column(data_pd, "Hypophonic voice")

    # Categorize Vocal tremor,Cephalic tremor,Mandibular tremor,Sialorrhoea,Dysphagia,Hypophonic voice columns
    data_hc = categorize_column(data_hc, "Vocal tremor")
    data_hc = categorize_column(data_hc, "Cephalic tremor")
    data_hc = categorize_column(data_hc, "Mandibular tremor")
    data_hc = categorize_column(data_hc, "Sialorrhoea")
    data_hc = categorize_column(data_hc, "Dysphagia")
    data_hc = categorize_column(data_hc, "Hypophonic voice")

    # Clean diagnosis
    # Clean diagnosis to know the number of patients per condition
    data_pd["Diagnosis"] = (
        data_pd["Diagnosis"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.replace(" ", "-")
        .str.replace("de", "")
        .str.replace("--", "-")
        .str.rstrip("-")
    )

    # Fill na iwth "unknown"
    data_pd["Diagnosis"] = data_pd["Diagnosis"].fillna("Unknown")

    # IF the diagnosis starts with "enfermedad", we will create a new column named "Binary diagnosis" and will fill it with PD
    # idx_enfermedad = data_pd["Diagnosis"].str.lower().str.startswith("enfermedad")
    # data_pd.loc[idx_enfermedad, "Binary diagnosis"] = "PD"
    # # If the diagnosis starts with "sindrome", we will create a new column named "Binary diagnosis" and will fill it with PS
    # idx_sindrome = data_pd["Diagnosis"].str.lower().str.startswith("sindrome")
    # data_pd.loc[idx_sindrome, "Binary diagnosis"] = "PS"

    data_hc["Diagnosis"] = (
        data_hc["Diagnosis"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.replace(" ", "-")
        .str.replace("de", "")
        .str.replace("--", "-")
        .str.replace("controle", "")
        .str.replace("---", "-")
        .str.rstrip("-")
    )

    # Clean Fibro/CCVV column. This the text starts with "No", we will assume it is "No", so check if starts with No
    idx_no = data_pd["Fibro/CCVV"].str.lower().str.startswith("no ")
    data_pd.loc[idx_no, "Fibro/CCVV"] = "Not performed"
    # Now check if its start with "normales"
    idx_normales = data_pd["Fibro/CCVV"].str.lower().str.startswith("normales")
    data_pd.loc[idx_normales, "Fibro/CCVV"] = "Normal"
    # Now if its starts with CCVV
    idx_ccvv = data_pd["Fibro/CCVV"].str.lower().str.startswith("ccvv")
    data_pd.loc[idx_ccvv, "Fibro/CCVV"] = "Normal"
    idx_en = data_pd["Fibro/CCVV"].str.lower().str.startswith("en")
    data_pd.loc[idx_en, "Fibro/CCVV"] = "Normal"

    # Clean the same for healthy
    # First check nan and fill with "Not performed"
    data_hc["Fibro/CCVV"] = data_hc["Fibro/CCVV"].fillna("Not performed")
    # Now check if starts with No
    idx_no = data_hc["Fibro/CCVV"].str.lower().str.startswith("no ")
    data_hc.loc[idx_no, "Fibro/CCVV"] = "Not performed"
    # Now if its starts with CCVV
    idx_ccvv = data_hc["Fibro/CCVV"].str.lower().str.startswith("ccvv")
    data_hc.loc[idx_ccvv, "Fibro/CCVV"] = "Normal"

    # If starts with cvderecha
    idx_cv = data_hc["Fibro/CCVV"].str.lower().str.startswith("cvderecha")
    data_hc.loc[idx_cv, "Fibro/CCVV"] = "Right hypothonic vocal cord"

    # Clean labour situation column
    # Remove accents
    data_pd["Labour situation"] = (
        data_pd["Labour situation"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.replace(" ", "-")
        .str.rstrip("-")
    )
    # Make all labour situation to same gender
    data_pd["Labour situation"] = data_pd["Labour situation"].replace(
        "jubilado", "retired"
    )
    data_pd["Labour situation"] = data_pd["Labour situation"].replace(
        "jubilada", "retired"
    )
    data_pd["Labour situation"] = data_pd["Labour situation"].replace(
        "ama-de-casa", "housekeeper"
    )

    # Clean labour situation column for healthy
    # Remove accents
    data_hc["Labour situation"] = (
        data_hc["Labour situation"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.replace(" ", "-")
        .str.rstrip("-")
    )
    # Make all labour situation to same
    data_hc["Labour situation"] = data_hc["Labour situation"].replace(
        "jubilado", "retired"
    )
    data_hc["Labour situation"] = data_hc["Labour situation"].replace(
        "jubilada", "retired"
    )
    data_hc["Labour situation"] = data_hc["Labour situation"].replace(
        "ama-de-casa", "housekeeper"
    )
    data_hc["Labour situation"] = data_hc["Labour situation"].replace(
        "pas", "administrativo"
    )
    data_hc["Labour situation"] = data_hc["Labour situation"].replace(
        "jubilada,-exprofesora", "retired teacher"
    )
    data_hc["Labour situation"] = data_hc["Labour situation"].replace(
        "profresor", "professor"
    )
    data_hc["Labour situation"] = data_hc["Labour situation"].replace(
        "professor", "professor"
    )

    # Drop "Review" column
    data_pd = data_pd.drop(columns=["Review "])
    data_hc = data_hc.drop(columns=["Review "])

    # Save the dataframes
    data_pd.to_csv("data/data_pd.csv", index=False)
    data_hc.to_csv("data/data_hc.csv", index=False)


if __name__ == "__main__":
    main()
