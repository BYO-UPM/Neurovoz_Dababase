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
            df = df.rename(columns={column: model.translate(column, source_lang="es", target_lang="en")})
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
            df_translated = df_translated.rename(columns={df_translated.columns[0]: "text_patient_id"})

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
