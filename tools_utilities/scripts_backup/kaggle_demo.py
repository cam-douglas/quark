from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

api = KaggleApi()
api.authenticate()  # Auto-loads your ~/.kaggle/kaggle.json

# Download Titanic dataset
api.dataset_download_files('heptapod/titanic', path='./data', unzip=True)

# Load into Pandas
df = pd.read_csv('./data/train.csv')
print(df.head())