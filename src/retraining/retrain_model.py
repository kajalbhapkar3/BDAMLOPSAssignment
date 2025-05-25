import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def retrain():
    print("♻️ Retraining model...")
    df = pd.read_csv("data/processed/train.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)

    joblib.dump(model, "app/model.pkl")
    print("✅ Model retrained and saved.")

if __name__ == "__main__":
    retrain()
