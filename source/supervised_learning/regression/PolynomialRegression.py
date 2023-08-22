import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Input data
X = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]).reshape((-1, 1))
# Output data
y = np.array([10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55])

# Transform input data
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Fit the model
model = LinearRegression()
model.fit(X_poly, y)

# Visualize the results
plt.scatter(X, y)
plt.plot(X, model.predict(poly.fit_transform(X)), color='r')
plt.show()
