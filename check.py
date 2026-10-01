import pandas as pd
import sklearn
import flask

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
print("scikit-learn:", sklearn.__version__)
print("flask:", flask.__version__)
print("ขนาดข้อมูล:", df.shape)