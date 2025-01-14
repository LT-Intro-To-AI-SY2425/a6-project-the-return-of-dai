import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("banana_quality.csv")
data['Quality'].replace(['Good','Bad'],[1,0],inplace=True)

x = data[["Weight", "Softness", "Ripeness"]].values
y = data["Quality"].values

scaler = StandardScaler().fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y)

model = linear_model.LogisticRegression().fit(x_train,y_train)

print("Accuracy: " + str(model.score(x_test, y_test)))