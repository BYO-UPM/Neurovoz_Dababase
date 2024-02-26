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
    print(f"Number of healthy controls: {len(df_hc)}")
    print(f"Number of patients: {len(df_pd)}")
    print(f"Number of total samples: {len(df_hc) + len(df_pd)}")
    # print mean and std of audios per patient
    df_total = pd.concat([df_hc, df_pd])
    print(f"Mean number of audios TOTAL: {np.mean(df_total.groupby('ID').size())}")
    print(f"Std dev of number of audios TOTAL: {np.std(df_total.groupby('ID').size())}")
    # Print number of audios per patient in mean and std
    print("=============== Healthy Controls ===============")
    print(f"Mean number of audios per patient: {np.mean(df_hc.groupby('ID').size())}")
    print(
        f"Std dev of number of audios per patient: {np.std(df_hc.groupby('ID').size())}"
    )
    print("=============== Patients ===============")
    print(f"Mean number of audios per patient: {np.mean(df_pd.groupby('ID').size())}")
    print(
        f"Std dev of number of audios per patient: {np.std(df_pd.groupby('ID').size())}"
    )
    # Print number of unique patients
    print("=============== Unique patients ===============")
    print(f"Number of unique healthy controls: {len(df_hc['ID'].unique())}")
    print(f"Number of unique patients: {len(df_pd['ID'].unique())}")
    print(
        f"Number of unique patients: {len(df_hc['ID'].unique()) + len(df_pd['ID'].unique())}"
    )

    # analyse the number of missing data per column
    print("=============== Missing data ===============")
    print("Missing data in healthy controls")
    print(df_hc.isnull().sum())
    print("Missing data in patients")
    print(df_pd.isnull().sum())
    # In percentage
    print("=============== Missing data in percentage ===============")
    print("Missing data in healthy controls")
    print(df_hc.isnull().sum() / len(df_hc))
    print("Missing data in patients")
    print(df_pd.isnull().sum() / len(df_pd))

    # Plot the missingness of the data
    import missingno as msno
    import matplotlib.pyplot as plt

    msno.matrix(df_hc.groupby(["ID"]).first())
    plt.savefig("missing_data_hc.png")
    msno.matrix(df_pd.groupby(["ID"]).first())
    plt.savefig("missing_data_pd.png")

    # The columns "Vocal tremor',
    #   'Cephalic tremor', 'Mandibular tremor', 'Sialorrhoea', 'Dysphagia',
    #   'Hypophonic voice'
    # are binary, propose a plot to show their distribution
    print("=============== Binary columns ===============")
    binary_columns = [
        "Vocal tremor",
        "Cephalic tremor",
        "Mandibular tremor",
        "Sialorrhoea",
        "Dysphagia",
        "Hypophonic voice",
    ]
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming `df` is your DataFrame
    sns.heatmap(
        df_pd[binary_columns].T, cmap="YlGnBu", cbar_kws={"label": "Presence/Absence"}
    )
    plt.title("Heatmap of Binary Indicators")
    plt.ylabel("Indicator")
    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.show()


    # Number of columsn in the dataset
    print("=============== Number of columns ===============")
    print(f"Number of columns in the healthy dataset: {len(df_hc.columns)}")
    print(f"Number of columns in the patient dataset: {len(df_pd.columns)}")
    


def main():
    general_statistic()


if __name__ == "__main__":
    main()