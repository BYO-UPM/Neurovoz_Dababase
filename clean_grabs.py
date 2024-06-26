import pandas as pd
from easynmt import EasyNMT

# Initialize the translator
model = EasyNMT("opus-mt")


# Function to translate a dataframe's content from Spanish to English
def translate_df(df):
    # Translate each cell in the dataframe, skipping the first column 6 columns (id and grabs columns)
    for column in df.columns[6:]:
        if (
            df[column].dtype == object and column != "Total"
        ):  # Check if the column is of type 'object', indicating it could be string text
            print("Translating column:", column)
            df[column] = df[column].apply(
                lambda x: (
                    model.translate(x, source_lang="es", target_lang="en")
                    if pd.notnull(x)
                    else x
                )
            )
            # Translate the column name
            df = df.rename(
                columns={
                    column: model.translate(column, source_lang="es", target_lang="en")
                }
            )
    return df


def process_excel(input_excel_path, output_excel_path):
    # Load the Excel file
    with pd.ExcelFile(input_excel_path) as xls:
        # List to hold dataframes
        dfs = []

        # Iterate through each sheet
        for sheet_name in xls.sheet_names:
            print("=========================================================")
            print("Processing sheet:", sheet_name)
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Remove 'Clinic' column if exists
            if "CLINICA" in df.columns:
                print("removing clinic column...")
                df = df.drop(columns=["CLINICA"])

            # Translate the dataframe
            df_translated = translate_df(df)

            # Rename first column to "text_patient_id"
            df_translated = df_translated.rename(
                columns={df_translated.columns[0]: "text_patient_id"}
            )

            print("removing unnamed columns...")
            # Remove unnamed columns
            df_translated = df_translated.loc[
                :, ~df_translated.columns.str.contains("^Unnamed")
            ]

            print("appending")
            # Append the processed dataframe
            dfs.append((sheet_name, df_translated))

    # Save each sheet in a separate .csv file with the same name as the sheet
    for sheet_name, df in dfs:
        output_name = "data/" + sheet_name + ".csv"
        df.to_csv(output_name, index=False)
        print("Saved:", output_name)


# Example usage
input_excel_path = "data/Evaluación Janaína Mendes .xlsx"
output_excel_path = "data/grbas.xlsx"
process_excel(input_excel_path, output_excel_path)


import pandas as pd
import os


def clean_string(s):
    return (
        s.str.replace(r"[^\w\s]", "")
        .str.replace(" ", "")
        .str.replace(".", "")
        .str.lower()
    )


# read gRABS features

path_folder = "data/version_to_zenodo/grbas"

# Read all csv files
for file in os.listdir(path_folder):
    if file.endswith(".csv"):
        print("Reading:", file)
        df = pd.read_csv(os.path.join(path_folder, file))

        # IF "text_patient_id" is NAN, drop that row
        df = df.dropna(subset=["text_patient_id"])

        # Converts columns GRBAS to float
        for column in ["G", "R", "B", "A", "S", "TOTAL"]:
            # Substitute "-" for "nan"
            if "-" in df[column].values:
                df[column] = df[column].str.replace("-", "nan")
            df[column] = df[column].astype(float)

        # Save the cleaned dataframe
        output_name = path_folder + "/" + file
        df.to_csv(output_name, index=False)
        print("Saved:", output_name)

        # Clean "GLOTOTIC ATTACK" if the column exists
        if "GLOTOTIC ATTACK" not in df.columns:
            print("Column 'GLOTOTIC ATTACK' not found in the dataframe")
            continue
        else:

            # First, clean all puntuaction (remove "." and similar)
            # then, clean blank spaces and write all in lower case
            df["GLOTOTIC ATTACK"] = df["GLOTOTIC ATTACK"].apply(
                lambda x: clean_string(pd.Series([x])).iloc[0] if pd.notna(x) else x
            )

        # Clean "TONO" if the column exists
        if "TONO" not in df.columns:
            print("Column 'TONO' not found in the dataframe")
            continue
        else:
            # First, clean all puntuaction (remove "." and similar)
            # then, clean blank spaces and write all in lower case
            df["TONE"] = df["TONE"].apply(
                lambda x: clean_string(pd.Series([x])).iloc[0] if pd.notna(x) else x
            )
            # Substitute "acute" for high-pitched
            df["TONE"] = df["TONE"].str.replace("acute", "high-pitched")
            df["TONE"] = df["TONE"].str.replace("agudi", "high-pitched")
            df["TONE"] = df["TONE"].str.replace("agudio", "high-pitched")
            df["TONE"] = df["TONE"].str.replace("high-pitchedo", "high-pitched")

            # Substitue "severe" for deep
            df["TONE"] = df["TONE"].str.replace("severe", "deep")

            # drop TONO
            df = df.drop(columns=["TONO"])

        # CLEAN "QUALITY FONATION"
        if "QUALITY FONATION" not in df.columns:
            print("Column 'QUALITY FONATION' not found in the dataframe")
            continue
        else:
            # First, clean all puntuaction (remove "." and similar)
            # then, clean blank spaces and write all in lower case
            df["QUALITY PHONATION"] = df["QUALITY FONATION"].apply(
                lambda x: clean_string(pd.Series([x])).iloc[0] if pd.notna(x) else x
            )

            # drop QUALITY FONATION
            df = df.drop(columns=["QUALITY FONATION"])

        # Clean "INTENSITY"
        if "INTENSITY" not in df.columns:
            print("Column 'INTENSITY' not found in the dataframe")
            continue
        else:
            # First, clean all puntuaction (remove "." and similar)
            # then, clean blank spaces and write all in lower case
            df["INTENSITY"] = df["INTENSITY FONATION"].apply(
                lambda x: clean_string(pd.Series([x])).iloc[0] if pd.notna(x) else x
            )

        # Clean speed
        if "SPEED" not in df.columns:
            print("Column 'SPEED' not found in the dataframe")
            continue
        else:
            # First, clean all puntuaction (remove "." and similar)
            # then, clean blank spaces and write all in lower case
            df["SPEED"] = df["SPEED"].apply(
                lambda x: clean_string(pd.Series([x])).iloc[0] if pd.notna(x) else x
            )
        # Clean RESONANCE
        if "RESONANCE" not in df.columns:
            print("Column 'RESONANCE' not found in the dataframe")
            continue
        else:
            # First, clean all puntuaction (remove "." and similar)
            # then, clean blank spaces and write all in lower case
            df["RESONANCE"] = df["RESONANCE"].apply(
                lambda x: clean_string(pd.Series([x])).iloc[0] if pd.notna(x) else x
            )

        # clean "INTELIGIBILITY"
        if "INTELIGIBILITY" not in df.columns:
            print("Column 'INTELIGIBILITY' not found in the dataframe")
            continue
        else:
            # First, clean all puntuaction (remove "." and similar)
            # then, clean blank spaces and write all in lower case
            df["INTELLIGIBILITY"] = df["INTELLIGIBILITY"].apply(
                lambda x: clean_string(pd.Series([x])).iloc[0] if pd.notna(x) else x
            )

            # Subnstitue "1normal" for "normal"
            df["INTELLIGIBILITY"] = df["INTELLIGIBILITY"].str.replace(
                "1normal", "normal"
            )

            # Substityte 2milddeficiency to 2 mild deficiency
            df["INTELLIGIBILITY"] = df["INTELLIGIBILITY"].str.replace(
                "2milddeficiency", "2 mild deficiency"
            )

            # Substitute 3moderatedeficiencyto 3 moderate deficiency
            df["INTELLIGIBILITY"] = df["INTELLIGIBILITY"].str.replace(
                "3moderatedeficiency", "3 moderate deficiency"
            )

            # Substitute 4severemoderatedeficiency to 4 severe moderate deficiency
            df["INTELLIGIBILITY"] = df["INTELLIGIBILITY"].str.replace(
                "4severemoderatedeficiency", "4 severe moderate deficiency"
            )

            # Substitute 5severedeficiency to 5 severe deficiency
            df["INTELLIGIBILITY"] = df["INTELLIGIBILITY"].str.replace(
                "5severedeficiency", "5 severe deficiency"
            )

            # Subtitute milddeficiency to 2 mild deficiency
            df["INTELLIGIBILITY"] = df["INTELLIGIBILITY"].str.replace(
                "milddeficiency", "2 mild deficiency"
            )

            df = df.drop(columns=["INTELIGIBILITY"])
        # clean "PROSODIA"
        if "PROSODIA" not in df.columns:
            print("Column 'PROSODIA' not found in the dataframe")
            continue
        else:
            # First, clean all puntuaction (remove "." and similar)
            # then, clean blank spaces and write all in lower case
            df["PROSODY"] = (
                df["PROSODIA"]
                .str.replace(r"[^\w\s]", "")
                .str.replace(" ", "")
                .str.replace(".", "")
                .str.lower()
            )

            # Drop
            df = df.drop(columns=["PROSODIA"])
            # Drop

        # Save the cleaned dataframe
        output_name = path_folder + "/" + file
        df.to_csv(output_name, index=False)
        print("Saved:", output_name)


# Read all grabs and extract text_patient_id and GRBAS values
path_folder = "data/version_to_zenodo/grbas"

f_df_pd = pd.DataFrame()
f_df_hc = pd.DataFrame()

# Read all csv files
for file in os.listdir(path_folder):
    if file.endswith(".csv"):
        print("Reading:", file)
        df = pd.read_csv(os.path.join(path_folder, file))

        # IF "text_patient_id" is NAN, drop that row
        df = df.dropna(subset=["text_patient_id"])

        # Extract "text_patient_id" and "GRBAS" columns
        df = df[["text_patient_id", "G", "R", "B", "A", "S", "TOTAL"]]
        df["ID"] = (
            df["text_patient_id"]
            .str.split("_")
            .str[-1]
            .str.split(".")
            .str[0]
            .astype(int)
        )

        # Read form metadata data_hc and data_pd and get "ID" column to match who is PD and who is HC
        # Read the metadata
        hc = pd.read_csv("data/version_to_zenodo/metadata/metadata_hc.csv")
        park = pd.read_csv("data/version_to_zenodo/metadata/metadata_pd.csv")

        # From df, create a df_pd and df_hc where id matches
        df_pd = df[df["ID"].isin(park["ID"])]
        df_hc = df[df["ID"].isin(hc["ID"])]

        # Append to the final dataframes
        f_df_pd = pd.concat([f_df_pd, df_pd])
        f_df_hc = pd.concat([f_df_hc, df_hc])

# Asser that none ID is in both dataframes
assert f_df_pd["ID"].isin(f_df_hc["ID"]).sum() == 0


# Do five plots, one per each GRBAS value. In each plot, plot the histogram of HC versus the histogram of PD
import matplotlib.pyplot as plt
import numpy as np

# Create a figure with a single row of subplots
fig, axs = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

# Define the width of the bars
bar_width = 0.4

# Iterate over each GRBAS value
for i, grbas_value in enumerate(["G", "R", "B", "A", "S"]):
    # Define the bins
    bins = np.arange(-0.5, 4, 1)  # Assuming GRBAS values go from 0 to 3

    # Plot the histogram of HC values
    axs[i].hist(
        f_df_hc[grbas_value].astype(float),
        bins=bins - bar_width / 2,
        alpha=0.5,
        label="HC",
        color="blue",
        width=bar_width,
        align="left",
    )

    # Plot the histogram of PD values
    axs[i].hist(
        f_df_pd[grbas_value].astype(float),
        bins=bins + bar_width / 2,
        alpha=0.5,
        label="PD",
        color="red",
        width=bar_width,
        align="left",
    )

    # Set the title
    axs[i].set_title(f"{grbas_value}")

    # SEt x_ticks
    axs[i].set_xticklabels(["0", "0", "1", "2", "3"])

    axs[i].legend()

# Set the y-axis label
axs[0].set_ylabel("Frequency")

plt.tight_layout()
# Save in dpi=300
plt.savefig("data/grbas_histograms.png", dpi=300)


# Annonymze date
# Define the function to anonymize the surgery date
from datetime import datetime

df_hc = pd.read_csv("data/version_to_zenodo/metadata/metadata_hc.csv")
df_pd = pd.read_csv("data/version_to_zenodo/metadata/metadata_pd.csv")


def anonymize_surgery_date(date_str):
    try:
        # First check if it is NaN, in that case, just skip
        if pd.isna(date_str):
            return date_str
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.replace(day=1).strftime("%Y-%m-%d")
    except ValueError:
        return date_str


# Clean all dates
df_hc["Date "] = df_hc["Date "].apply(anonymize_surgery_date)
df_hc["Date Evaluation Scales "] = df_hc["Date Evaluation Scales "].apply(
    anonymize_surgery_date
)

df_pd["Date "] = df_pd["Date "].apply(anonymize_surgery_date)
df_pd["Date Evaluation Scales "] = df_pd["Date Evaluation Scales "].apply(
    anonymize_surgery_date
)

# Save back csv
df_hc.to_csv("data/version_to_zenodo/metadata/metadata_hc.csv", index=False)
df_pd.to_csv("data/version_to_zenodo/metadata/metadata_pd.csv", index=False)


# Get all grabs files

path_folder = "data/version_to_zenodo/grbas"
grbas_df = pd.DataFrame()

# Read all csv files
for file in os.listdir(path_folder):
    if file.endswith(".csv"):
        print("Reading:", file)
        df = pd.read_csv(os.path.join(path_folder, file))

        # IF "text_patient_id" is NAN, drop that row
        df = df.dropna(subset=["text_patient_id"])

        # Extract "text_patient_id" and "GRBAS" columns
        df = df[["text_patient_id", "G", "R", "B", "A", "S", "TOTAL"]]
        df["ID"] = (
            df["text_patient_id"]
            .str.split("_")
            .str[-1]
            .str.split(".")
            .str[0]
            .astype(int)
        )

        # Pd concat
        grbas_df = pd.concat([grbas_df, df])

path_folder_2 = "data/version_to_zenodo/audios"

# Get a list of all audio files
audio_files = []
ids = []
for file in os.listdir(path_folder_2):
    if file.endswith(".wav"):
        audio_files.append(file)
        ids.append(file[3:])


# Get the files for which we have audio but not GRBAS, tha tis, ids exists but not in grbas_df["text_patient_id"]
missing_grbas = set(ids) - set(grbas_df["text_patient_id"])
# The other way arround, we have GRBAS but not audio
missing_audio = set(grbas_df["text_patient_id"]) - set(ids)


# For all missing grabs, generate:
# 1. A new dataframe with same structure as grbas_df with the missing ids and emtpy values
# 2. Save the dataframe to a csv file with hte name "data/version_to_zenodo/grbas/missing_grbas.csv"
# 3. Generate an audio folder, named missing_grbas, with the same structure as the other audio folders witht he audios for which we dont have grabs
# 4. Save the audio folder to "data/version_to_zenodo/audios/missing_grbas"

# 1. Generate the new dataframe
missing_grbas_df = pd.DataFrame()
missing_grbas_df["text_patient_id"] = list(missing_grbas)
missing_grbas_df["G"] = np.nan
missing_grbas_df["R"] = np.nan
missing_grbas_df["B"] = np.nan
missing_grbas_df["A"] = np.nan
missing_grbas_df["S"] = np.nan
missing_grbas_df["TOTAL"] = np.nan
missing_grbas_df["ID"] = (
    missing_grbas_df["text_patient_id"]
    .str.split("_")
    .str[-1]
    .str.split(".")
    .str[0]
    .astype(int)
)

# Save the dataframe
missing_grbas_df.to_csv("data/version_to_zenodo/missing_grbas.csv", index=False)

# Get that audios and copy them into a new folder
import shutil

# Create the folder
os.makedirs("data/version_to_zenodo/audios/missing_grbas", exist_ok=True)

# Copy the files
for file in audio_files:
    if file[3:] in missing_grbas:
        shutil.copy(
            os.path.join(path_folder_2, file),
            os.path.join("data/version_to_zenodo/audios/missing_grbas", file),
        )
