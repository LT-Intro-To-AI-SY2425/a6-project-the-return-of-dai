import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations

data = pd.read_csv("banana_quality.csv")
data["Quality"].replace(["Good", "Bad"], [1, 0], inplace=True)

models = ["Size", "Weight", "Sweetness", "Softness", "HarvestTime", "Ripeness", "Acidity"]
combinations = [list(combo) for combo in combinations(models, 2)]

best_score = 0
best_features = None
top_models = []

for i, features in enumerate(combinations, 1):
    print(f"Testing model {i}")
    x = data[features].values
    y = data["Quality"].values

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = linear_model.LogisticRegression().fit(x_train, y_train)
    score = model.score(x_test, y_test)
    top_models.append((features, score))

    if score > best_score:
        best_score = score
        best_features = features

top_models.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 models:")
for i, (features, score) in enumerate(top_models[:5], 1):
    print(f"{i}. Model: {features}, R^2 = {score:.2f}")

print(f"\nBest model: {best_features}, R^2 = {best_score:.2f}")