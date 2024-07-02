import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def arrhenius_model(T, A, Ea):
    """
    Arrhenius equation model function to calculate rate constants based on
    temperature.

    Parameters:
    T (np.array): Temperatures at which rate constants are measured
    (in degrees Celsius).
    A (float): Pre-exponential factor.
    Ea (float): Activation energy in J/mol.

    Returns:
    np.array: Computed rate constants for given temperatures.
    """
    R = 8.314  # Gas constant in J/(mol*K)
    T = T + 273  # Convert temperature from Celsius to Kelvin
    return A * np.exp(-Ea / (R * T))


# Experimental data
T_expt = np.array([477, 523, 577, 623])  # Temperatures in Celsius
k_expt = np.array([0.0018, 0.0027, 0.030, 0.26])  # Rate constants

# Initial guesses for the Arrhenius parameters
initial_guess = [1.39406453358858e15, 271.867e3]

# Fitting the Arrhenius model to experimental data using non-linear
# least squares
popt, pcov = curve_fit(arrhenius_model, T_expt, k_expt, p0=initial_guess)
A_opt, Ea_opt = popt  # Optimized parameters

# Print the optimized parameters
print(f"Optimized A: {A_opt:.2e}, Ea: {Ea_opt:.2e} J/mol")

# Generating values for plotting the fitted model
T_fit = np.linspace(min(T_expt), max(T_expt), 100)
# Temperature range for the fit
k_fit = arrhenius_model(T_fit, *popt)
# Rate constants calculated with the fitted model

# Plotting the experimental data and the fitted curve
plt.figure(figsize=(8, 5))
plt.scatter(T_expt, k_expt, color="red", label="Experimental Data")
plt.plot(T_fit, k_fit, label="Fitted Arrhenius Model", color="blue")
plt.xlabel("Temperature (°C)")
plt.ylabel("Rate constant (1/s)")
plt.title("Fit of Arrhenius Model to Experimental Data")
plt.legend()
plt.show()
