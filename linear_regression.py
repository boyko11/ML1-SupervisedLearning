import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([1, 3, 6, 10, 11, 13])
X = X.reshape(X.shape[0], 1)
y = np.array([1, 0, 5, 2, 1, 4])
y = y.reshape(y.shape[0], 1)
print(X.shape)
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)
