import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
warnings.simplefilter('ignore')

df = pd.read_csv("data/Mobile Price Prediction Datatset.csv")
original_df = df.copy()

missing_cols = sorted(df.columns[df.isnull().any()].tolist(), key=lambda col: df[col].isnull().sum(), reverse=True)
num_rows = df.shape[0]


random_sample_cols = [col for col in missing_cols if df[col].isnull().sum() / num_rows < 0.06]
adding_cat_cols = [col for col in missing_cols if df[col].isnull().sum() / num_rows >= 0.06]
print(random_sample_cols)
print(adding_cat_cols)

# random sample imputation
for col in random_sample_cols:
    random_sample_train = df[col].dropna().sample(df[col].isnull().sum(), random_state=0)
    random_sample_train.index = df[df[col].isnull()].index
    df.loc[df[col].isnull(), col] = random_sample_train

mean_value = df['Selfi_Cam'].mean()
df['Selfi_Cam'].fillna(mean_value, inplace=True)

cols_to_scale = [col for col in df.columns if col != 'Price']
df[cols_to_scale].describe()

df.drop('Brand me', axis=1, inplace=True)

df.to_csv('data/processed_train.csv', index=False)