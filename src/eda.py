import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/credit_risk.csv"
OUTPUT_DIR = "outputs"


def save_loan_status_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="loan_status")
    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/loan_status_distribution.png", dpi=150)
    plt.close()


def save_numeric_boxplot(df, column_name):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="loan_status", y=column_name)
    plt.title(f"{column_name} by Loan Status")
    plt.xlabel("Loan Status")
    plt.ylabel(column_name)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{column_name}_by_loan_status.png", dpi=150)
    plt.close()


def save_default_rate_by_category(df, column_name):
    default_rate = (
        df.groupby(column_name)["loan_status"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 4))
    default_rate.plot(kind="bar")
    plt.title(f"Default Rate by {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Default Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/default_rate_by_{column_name}.png", dpi=150)
    plt.close()

    return default_rate


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("Dataset shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn info:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False))

    print("\nTarget distribution:")
    print(df["loan_status"].value_counts())

    print("\nDefault rate:")
    print(df["loan_status"].value_counts(normalize=True))

    print("\nNumeric summary:")
    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length"
    ]
    print(df[numeric_cols].describe())

    print("\nAverage numeric values by loan_status:")
    print(df.groupby("loan_status")[numeric_cols].mean())

    print("\nDefault rate by home ownership:")
    print(save_default_rate_by_category(df, "person_home_ownership"))

    print("\nDefault rate by loan intent:")
    print(save_default_rate_by_category(df, "loan_intent"))

    print("\nDefault rate by loan grade:")
    print(save_default_rate_by_category(df, "loan_grade"))

    print("\nDefault rate by previous default record:")
    print(save_default_rate_by_category(df, "cb_person_default_on_file"))

    save_loan_status_distribution(df)

    for col in [
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "person_emp_length"
    ]:
        save_numeric_boxplot(df, col)

    print("\nEDA charts saved to outputs/:")
    print("- loan_status_distribution.png")
    print("- person_income_by_loan_status.png")
    print("- loan_amnt_by_loan_status.png")
    print("- loan_int_rate_by_loan_status.png")
    print("- loan_percent_income_by_loan_status.png")
    print("- person_emp_length_by_loan_status.png")
    print("- default_rate_by_person_home_ownership.png")
    print("- default_rate_by_loan_intent.png")
    print("- default_rate_by_loan_grade.png")
    print("- default_rate_by_cb_person_default_on_file.png")


if __name__ == "__main__":
    main()