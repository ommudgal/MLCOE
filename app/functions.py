def load_data(path, rows=10):
    import pandas as pd

    data = pd.read_csv(path, nrows=rows)
    return data
