import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("banana_quality.csv")
data["Quality"].replace(["Good", "Bad"], [1, 0], inplace=True)

# Top 4 models based on R^2 (see daeval.py)
models = [["Sweetness", "HarvestTime"], ["Size", "Sweetness"], ["Weight", "HarvestTime"], ["HarvestTime", "Ripeness"]]

for features in models:
    x = data[features].values
    y = data["Quality"].values

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    model = linear_model.LogisticRegression().fit(x_train, y_train)
    print(f"Model: {features}, R^2 = {model.score(x_test, y_test):.2f}")

    good = data[data['Quality'] == 1]
    bad = data[data['Quality'] == 0]
    plt.scatter(good[features[1]], good[features[0]], color="g", label="Good")
    plt.scatter(bad[features[1]], bad[features[0]], color="r", label="Bad")
    plt.xlabel(features[1])
    plt.ylabel(features[0])
    plt.title(f"Banana Quality by {features[0]} and {features[1]}")
    x_values = np.linspace(-2, 2, 100)
    y_values = -(model.coef_[0][0] * x_values + model.intercept_) / model.coef_[0][1]
    plt.plot(x_values, y_values, label="Regression Line")
    plt.annotate(f"R^2 = {model.score(x_test, y_test):.2f}", xy=(0, 0), xytext=(0.01, 0.95), textcoords='axes fraction')
    plt.legend()
    plt.show()
