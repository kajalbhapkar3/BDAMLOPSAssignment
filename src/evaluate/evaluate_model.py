import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluate():
    print("📊 Evaluating model...")
    df = pd.read_csv("data/processed/train.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("app/model.pkl")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate()
