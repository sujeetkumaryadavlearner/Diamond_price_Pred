import pandas as pd


def ingest_data()->pd.DataFrame:
    data= pd.read_csv("/home/sujeet-kumar-yadav/ml_project/data/raw/diamonds .csv")
    return data
