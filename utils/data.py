import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_arsenal_data():
    df = pd.read_csv("data/matches.csv")

    df = df[["HoAw", "ArsenalScore", "OpponentScore"]].dropna()

    # win label
    df["ArsenalWin"] = (df["ArsenalScore"] > df["OpponentScore"]).astype(int)
    # feautre 1
    df["HomeGame"] = (df["HoAw"] == "home").astype(int)
    X = df[["HomeGame", "OpponentScore"]].values
    y = df["ArsenalWin"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    # scaling the feature so one isnt way bigger than the other
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test