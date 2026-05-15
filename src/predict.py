import joblib
import pandas as pd


MODEL_PATH = "outputs/model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_single_applicant(model, applicant_data):
    df = pd.DataFrame([applicant_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    label = "High Risk" if prediction == 1 else "Low Risk"

    if probability >= 0.7:
        risk_level = "High"
    elif probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return label, probability, risk_level


def main():
    model = load_model()

    sample_applicant = {
        "person_age": 24,
        "person_income": 30000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "D",
        "loan_amnt": 12000,
        "loan_int_rate": 16.5,
        "loan_percent_income": 0.40,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 3
    }

    label, probability, risk_level = predict_single_applicant(
        model,
        sample_applicant
    )

    print("Sample applicant prediction:")
    print(f"Predicted Risk: {label}")
    print(f"Default Probability: {probability:.4f}")
    print(f"Risk Level: {risk_level}")


if __name__ == "__main__":
    main()