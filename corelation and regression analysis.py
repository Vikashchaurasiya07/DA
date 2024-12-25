import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris

# Simple data generation for correlation and linear regression
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + np.random.randn(100) * 2

# Scatter plot & correlation
plt.scatter(X, y)
plt.title('Scatter Plot')
plt.show()
print("Correlation Coefficient:", np.corrcoef(X.squeeze(), y)[0, 1])

# Linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print model evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Logistic regression on Iris dataset
iris = load_iris()
X_iris = iris.data[:, :2]
y_iris = iris.target
log_reg = LogisticRegression().fit(X_iris, y_iris)

# Plotting decision boundary
x_min, x_max = X_iris[:, 0].min()-1, X_iris[:, 0].max()+1
y_min, y_max = X_iris[:, 1].min()-1, X_iris[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris)
plt.title('Logistic Regression (Iris)')
plt.show()
