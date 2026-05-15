from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train_model import train_models
from src.evaluation import evaluate_models
from src.visualization import (
    plot_model_comparison,
    plot_confusion_matrix,  
    plot_pca,
    plot_feature_importance
)

df = load_data("dataset//student_learning_air_quality.csv")

X, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

trained_models = train_models(X_train_scaled, y_train)

results_df, reports = evaluate_models(trained_models, X_test_scaled, y_test)

print(results_df)

with open("outputs/classification_reports.txt", "w") as f:
    f.write(reports)

plot_model_comparison(results_df)
plot_pca(X_train_scaled, y_train)

plot_confusion_matrix(
    trained_models["Random Forest"],
    X_test_scaled,
    y_test,
    "confustion_matrix_rf",
    "Random Forest Classifier Confusion Matrix"
)

plot_feature_importance(trained_models["Random Forest"], X.columns)


