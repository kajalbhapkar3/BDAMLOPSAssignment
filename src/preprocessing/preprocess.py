import pandas as pd
import os

def preprocess():
    print("⚙️ Preprocessing data...")
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv("data/raw/train.csv")
    df.fillna(0, inplace=True)  # Example: fill missing values
    df.to_csv("data/processed/train.csv", index=False)
    print("✅ Preprocessing completed. Saved to data/processed/train.csv")

if __name__ == "__main__":
    preprocess()