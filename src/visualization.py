import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA

def plot_model_comparison(results_df):
    results_df.plot(
        x="Model",
        y=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],
        kind="bar",
        figsize=(10,6)
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png")
    plt.close()

def plot_confusion_matrix(model, X_test_scaled, y_test, filename, title):
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test_scaled,
        y_test
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}.png")
    plt.close()

def plot_pca(X_trained_scaled, y_train):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_trained_scaled)

    plt.figure(figsize=(10,6))
    sns.scatterplot(
        x=X_pca[:,0],
        y=X_pca[:,1],
        c=y_train,
        cmap="viridis",
        alpha=0.7
    )
    plt.title("PCA of Training Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig("outputs/pca_scatter.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    importance_df=(
        importance_df.sort_values(by="Importance", ascending=False).head(10)
    )

    plt.figure(figsize=(10,6))

    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

