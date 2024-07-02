from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def generate_and_fit_model():
    """
    Generate synthetic data, fit a linear regression model, and print
    and plot the results.

    This function:
    - Generates synthetic data for regression analysis.
    - Fits a linear regression model.
    - Prints the model coefficients, intercept, and R-squared value.
    - Plots the histogram of residuals, and scatter plots of residuals against
      actual and predicted values.
    """
    # Generate synthetic data
    X, y, coefficients = make_regression(
        n_samples=100, n_features=2, noise=10, coef=True, random_state=5
    )
    # random_state ensures reproducibility of results

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Print model details
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("R-squared:", r2_score(y, y_pred))

    # Construct the equation of the regression line
    equation = f"y = {model.intercept_:.2f}"
    for i, coef in enumerate(model.coef_):
        equation += f" + {coef:.2f}X{i+1}"
    print("Equation of the line:", equation)

    # Compute residuals (actual - predicted)
    residuals = y - y_pred

    # Plotting the results in three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Histogram of residuals
    axs[0].hist(residuals, bins=20, color="skyblue", edgecolor="black")
    axs[0].set_title("Histogram of Residuals")
    axs[0].set_xlabel("Residuals")
    axs[0].set_ylabel("Frequency")

    # Scatter plot of residuals vs. actual values (y)
    axs[1].scatter(y, residuals, alpha=0.6, color="purple")
    axs[1].axhline(0, color="red", linewidth=2, linestyle="--")
    axs[1].set_title("Residuals vs. Actual Values (y)")
    axs[1].set_xlabel("Actual Values (y)")
    axs[1].set_ylabel("Residuals")

    # Scatter plot of residuals vs. predicted values (ŷ)
    axs[2].scatter(y_pred, residuals, alpha=0.6, color="green")
    axs[2].axhline(0, color="red", linewidth=2, linestyle="--")
    axs[2].set_title("Residuals vs. Predicted Values (ŷ)")
    axs[2].set_xlabel("Predicted Values (ŷ)")
    axs[2].set_ylabel("Residuals")

    plt.tight_layout()
    plt.show()


# Run the model generation and fitting function
if __name__ == "__main__":
    generate_and_fit_model()
