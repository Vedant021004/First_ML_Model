import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,r2_score
data = pd.read_csv("ml.csv")

print(data.head(5))

X = data[['YearsExperience']]
print(X)
Y = data["Salary"]
print(Y)


X_train,x_test,Y_train,y_test = train_test_split(X, Y, train_size = 0.2, random_state = 42)



model = LogisticRegression()

model.fit(X_train,Y_train) # trainign on 80% of the data

predict = model.predict(x_test) # prediction on 20% of the data

print(predict)

mean_absolute_error = mean_absolute_error(y_test,predict)

print(mean_absolute_error)

r2_score = r2_score(y_test, predict)

print(r2_score)

import pandas as pd

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predict
})

print(comparison)

import matplotlib.pyplot as plt

plt.scatter(y_test, predict)
plt.plot(y_test, predict)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig("graph.png")   
plt.show()