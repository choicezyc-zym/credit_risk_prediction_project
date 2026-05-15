import json
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/credit_risk.csv"
OUTPUT_DIR = "outputs"


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4)
    }

    report = classification_report(y_test, y_pred)

    return metrics, report, y_pred


def save_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()


def get_feature_names(preprocessor, numeric_features, categorical_features):
    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_feature_names = onehot.get_feature_names_out(categorical_features)

    return list(numeric_features) + list(cat_feature_names)


def save_logistic_feature_importance(model, feature_names):
    coefficients = model.named_steps["classifier"].coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })

    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()

    importance_df = importance_df.sort_values(
        by="abs_coefficient",
        ascending=False
    )

    importance_df.to_csv(
        f"{OUTPUT_DIR}/logistic_feature_importance.csv",
        index=False
    )

    top_features = importance_df.head(15).sort_values(
        by="coefficient",
        ascending=True
    )

    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"], top_features["coefficient"])
    plt.title("Top Logistic Regression Feature Coefficients")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/logistic_feature_importance.png", dpi=150)
    plt.close()


def save_random_forest_feature_importance(model, feature_names):
    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df.to_csv(
        f"{OUTPUT_DIR}/random_forest_feature_importance.csv",
        index=False
    )

    top_features = importance_df.head(15).sort_values(
        by="importance",
        ascending=True
    )

    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/random_forest_feature_importance.png", dpi=150)
    plt.close()


def train_models():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = load_data()

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )
    }

    all_metrics = {}

    best_model_name = None
    best_model = None
    best_f1 = -1

    for model_name, classifier in models.items():
        print("=" * 80)
        print(f"Training model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics, report, y_pred = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        all_metrics[model_name] = metrics

        print("Metrics:")
        print(metrics)

        print("\nClassification report:")
        print(report)

        save_confusion_matrix(y_test, y_pred, model_name)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = model_name
            best_model = pipeline

    all_metrics["best_model"] = best_model_name

    with open(f"{OUTPUT_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    joblib.dump(best_model, f"{OUTPUT_DIR}/model.pkl")

    print("=" * 80)
    print(f"Best model: {best_model_name}")

    fitted_preprocessor = best_model.named_steps["preprocessor"]

    feature_names = get_feature_names(
        fitted_preprocessor,
        numeric_features,
        categorical_features
    )

    if best_model_name == "logistic_regression":
        save_logistic_feature_importance(best_model, feature_names)

    if best_model_name == "random_forest":
        save_random_forest_feature_importance(best_model, feature_names)

    print("\nSaved outputs:")
    print("- outputs/model.pkl")
    print("- outputs/metrics.json")
    print("- outputs/confusion_matrix_logistic_regression.png")
    print("- outputs/confusion_matrix_random_forest.png")

    if best_model_name == "logistic_regression":
        print("- outputs/logistic_feature_importance.csv")
        print("- outputs/logistic_feature_importance.png")

    if best_model_name == "random_forest":
        print("- outputs/random_forest_feature_importance.csv")
        print("- outputs/random_forest_feature_importance.png")


if __name__ == "__main__":
    train_models()