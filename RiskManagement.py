import numpy as np
import pandas as pd
import random
from scipy.stats import norm

def historical_var(returns, confidence_level):
    # Calculate the historical value-at-risk
    var = np.percentile(returns, 100 - confidence_level * 100)
    return var

def monte_carlo_var(returns, confidence_level, simulations):
    # Generate random returns using Monte Carlo simulation
    simulated_returns = []
    for i in range(simulations):
        simulated_return = random.choices(returns, k=len(returns))
        simulated_returns.append(simulated_return)
        
    # Calculate the mean of each simulated return
    mean_simulated_returns = [np.mean(returns) for returns in simulated_returns]

    # Calculate the value-at-risk using the mean of the simulated returns
    var = np.percentile(mean_simulated_returns, 100 - confidence_level * 100)
    return var

def parametric_volatility_var(returns, confidence_level):
    # Calculate the mean and standard deviation of the returns
    mean = np.mean(returns)
    stddev = np.std(returns)

    # Calculate the z-score for the confidence level
    z_score = norm.ppf(confidence_level)

    # Calculate the value-at-risk using the mean and standard deviation of the returns
    var = mean - z_score * stddev

    return var

def risk_management(positions, returns, confidence_level, max_risk, method, simulations=1000):
    if method == "historical":
        # Calculate the historical value-at-risk
        var = historical_var(returns, confidence_level)
    elif method == "monte_carlo":
        # Calculate the Monte Carlo value-at-risk
        var = monte_carlo_var(returns, confidence_level, simulations)
    elif method == "parametric_volatility":
        # Calculate the parametric volatility value-at-risk
        var = parametric_volatility_var(returns, confidence_level)
    else:
        raise ValueError("Invalid method specified")

    # Calculate the portfolio value
    portfolio_value = positions.sum()

    # Calculate the maximum allowed risk
    max_allowed_risk = portfolio_value * max_risk

    # Check if the calculated value-at-risk exceeds the maximum allowed risk
    if var > max_allowed_risk:
        # Scale down the positions to meet the maximum risk constraint
        scale_factor = max_allowed_risk / var
        scaled_positions = positions * scale_factor
        return scaled_positions
    else:
        # Return the original positions if the value-at-risk is within bounds
        return positions

# Load the returns data into a pandas DataFrame
returns = pd.read_csv("returns.csv")

# Convert the returns data into a numpy array
returns = returns["returns"].values

# Define the positions in the portfolio
positions = np.array([100, 200, 300, 400])
# Define the confidence level
confidence_level = 0.95

# Define the maximum allowed risk as a fraction of portfolio value
max_risk = 0.1

# Define the method for calculating value-at-risk
method1 = "historical"
method2 = "monte_carlo"
method3 = "parametric_volatility"

# Apply the risk management model
scaled_positions_1 = risk_management(positions, returns, confidence_level, max_risk, method1)
scaled_positions_2 = risk_management(positions, returns, confidence_level, max_risk, method2)
scaled_positions_3 = risk_management(positions, returns, confidence_level, max_risk, method3)

# Print results
print("Scaled positions in historical VaR:", scaled_positions_1)
print("Scaled positions in Monte Carlo VaR:", scaled_positions_2)
print("Scaled positions in Parametric Volatility VaR:", scaled_positions_3)
