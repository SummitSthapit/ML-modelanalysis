import pandas as pd

from sklearn.metrics import (accuracy_score, classification_report,precision_score, recall_score, f1_score)

def evaluate_models(trained_models, X_test_scaled, y_test):
    results = []

    reports = ""

    for name, model in trained_models.items():
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        report = classification_report(y_test, predictions)

        reports += f"Classification Report for {name}:\n{report}\n\n"
        reports += report
        reports += "\n\n"

    results_df = pd.DataFrame(results)
    return results_df, reports

    