import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("iris.csv", delimiter=",")
data = data.drop(["petal.length", "petal.width"], axis=1)
data["variety"] = pd.Categorical(data["variety"]).codes
plt.scatter(data["sepal.length"], data["sepal.width"], c=data["variety"])
plt.xlabel("sepal.length")
plt.ylabel("sepal.width")
plt.show()