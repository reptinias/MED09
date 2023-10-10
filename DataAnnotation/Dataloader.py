import pandas as pd
from sklearn.model_selection import train_test_split


def loadDataset(datasetPath = "TestDataset.csv"):
    df = pd.read_csv(datasetPath, sep=";")
    df["Class"] = df["Class"].fillna(0)
    X = df["File"]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=10)

    return X_train, X_test, y_train, y_test

