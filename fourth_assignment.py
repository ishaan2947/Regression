import matplotlib.pyplot as plt
import numpy as np
from ml_uncertainty.non_linear_regression import NonLinearRegression


def arrhenius_model(T, coefs_):
    """
    Arrhenius equation model to calculate rate constants from temperature
    using given coefficients.

    Parameters:
    T (np.array): Array of temperatures at which rate constants are measured
                  (in degrees Celsius).
    coefs_ (tuple): Tuple containing the pre-exponential factor and
                    activation energy.

    Returns:
    np.array: Computed rate constants for given temperatures.
    """
    R = 8.314  # Universal gas constant in J/(mol*K)
    A, Ea = coefs_  # Unpacking the pre-exponential factor and
    # activation energy
    X = T[:, 0] + 273  # Convert temperature from Celsius to Kelvin
    k = A * np.exp(-Ea / (R * X))
    return k.reshape((-1,))


# Load experimental data
T_expt = np.array([477, 523, 577, 623]).reshape((-1, 1))
# Temperatures in Celsius
k_expt = np.array([0.0018, 0.0027, 0.030, 0.26])
# Corresponding rate constants

# Known true parameters for the model
true_params = np.array([1.39406453358858e15, 271.867e3])

# Fit the model using a nonlinear regression approach
nlr = NonLinearRegression(model=arrhenius_model, p0_length=2)
nlr.fit(T_expt, k_expt, p0=true_params * 0.5)
# Initial guess is 50% of true values

# Predict rate constants using both the fitted and true parameters
k_pred = nlr.predict(T_expt)
k_true_pred = nlr.predict(T_expt, params=true_params)

# Print fitted and true coefficients for comparison
print("Fitted coefficients:", nlr.coef_)
print("True coefficients:", true_params)

# Plotting experimental data, fitted model predictions,
# and true model predictions
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(T_expt.flatten(), k_expt, color="red", label="Experimental")
plt.plot(T_expt.flatten(), k_pred, color="blue", label="Fitted")
plt.plot(T_expt.flatten(), k_true_pred, color="green", label="True Params")
plt.legend()
plt.title("Comparison of Experimental vs. Predicted")

# Plot residuals to assess the fit
plt.subplot(122)
residuals = k_expt - k_pred  # Calculate residuals
plt.scatter(T_expt.flatten(), residuals, color="purple")
plt.axhline(y=0, color="r", linestyle="--")
plt.title("Residuals Plot")
plt.show()
