import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X, y, coefficients = make_regression(
    n_samples=100, n_features=2, noise=10, coef=True, random_state=5
)
# random_state allows results to be consist (can be any value I think)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R-squared:", r2_score(y, y_pred))

# For outputting the equation
# allows for the option of multiple features if needed
equation = f"y = {model.intercept_:.2f}"
for i, coef in enumerate(model.coef_):
    equation += f" + {coef:.2f}X{i+1}"

print("Equation of the line:", equation)
