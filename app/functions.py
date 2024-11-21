def load_data(path):
    import pandas as pd

    data = pd.read_csv(path)
    return data


with open(r"D:\Projects\MLCOE\Notebooks\models\Decision_Tree.joblib", "rb") as file:
    import joblib

    model = joblib.load(file)


def predict(model, data):
    return model.predict(data)
