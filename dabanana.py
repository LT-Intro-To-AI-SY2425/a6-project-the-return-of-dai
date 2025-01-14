import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TODO: compare more models (quality to weight/size, etc.)
# also maybe we should do one that's not multivariable?

data = pd.read_csv("banana_quality.csv")
data['Quality'].replace(['Good','Bad'], [1,0], inplace=True)

x = data[["Weight", "Ripeness"]].values
y = data["Quality"].values

scaler = StandardScaler().fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = linear_model.LogisticRegression().fit(x_train, y_train)

rsq = model.score(x_test, y_test)
print("Accuracy: " + str(rsq))

# Plot 3D figure + best fit plane + display equation and R^2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data["Weight"], data["Ripeness"], data["Quality"], c=data["Quality"])
ax.set_xlabel('Weight')
ax.set_ylabel('Ripeness')
ax.set_zlabel('Quality')

xx, yy = np.meshgrid(np.linspace(data["Weight"].min(), data["Weight"].max(), 10),
                     np.linspace(data["Ripeness"].min(), data["Ripeness"].max(), 10))
zz = model.predict_proba(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
zz = zz.reshape(xx.shape)

ax.plot_surface(xx, yy, zz, color='blue', alpha=0.5)

coef = model.coef_[0]
intercept = model.intercept_[0]
equation = f'Quality = {coef[0]:.2f} * Weight + {coef[1]:.2f} * Ripeness + {intercept:.2f}'
ax.text2D(0.05, 0.95, equation, transform=ax.transAxes)
ax.text2D(0.05, 0.90, f'R^2 = {rsq:.2f}', transform=ax.transAxes)

plt.show()
