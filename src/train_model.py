from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_models(X_train_scaled, y_train):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        print(f"{name} trained successfully.")

    return trained_models
    
    

