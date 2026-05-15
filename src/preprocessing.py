import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import(LabelEncoder, StandardScaler)

def preprocess_data(df):
    # Remove unnecessary columns
    df=df.drop(columns=['student_id'])

    #ordinal encodings:
    grade_mapping = {
    "Grade 6": 6,
    "Grade 7": 7,
    "Grade 8": 8,
    "Grade 9": 9,
    "Grade 10": 10,
    "Grade 11": 11,
    }
    df["grade"] = df["grade"].map(grade_mapping)

    air_quality_mapping = {
    "Poor": 0,
    "Moderate": 1,
    "Good": 2
    }
    df["air_quality_label"] = df["air_quality_label"].map(air_quality_mapping)

    # features and target
    X = df.drop("performance_label", axis=1)
    y = df["performance_label"]

    # One hot encoding for categorical features
    categorical_columns = [
    "day",
    "period",
    "subject"
    ]
    X = pd.get_dummies(X,columns=categorical_columns,drop_first=True)
    
    #Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=0.2,
        random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X,
        X_train_scaled,
        X_test_scaled, 
        y_train, 
        y_test
        )
        

