import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("ml.csv")

print(data.head(5))

X = data['YearsExperience']
print(X)
Y = data["Salary"]
print(Y)