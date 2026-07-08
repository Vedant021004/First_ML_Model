import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Use read_excel for .xlsx files
df = pd.read_excel('AI_Placement_Predictor_Dataset.xlsx')

print(df.head(10))

print(df.info())

print(df.describe())

print(df[df.isnull()].sum())

print(df[df.duplicated()].sum())

print(df.shape)