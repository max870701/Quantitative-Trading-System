import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def calculate_returns(prices):
    # Calculate the returns from the prices
    returns = prices.pct_change().dropna()
    return returns

def calculate_performance_metrics(portfolio_value):
    # Calculate the returns from the portfolio value
    returns = portfolio_value.pct_change().dropna()
    
    # Calculate the profit and loss
    pnl = portfolio_value.iloc[-1] - portfolio_value.iloc[0]
    
    # Calculate the annualized return
    num_years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365
    annualized_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / num_years) - 1
    
    # Calculate the Sharpe ratio
    daily_returns = returns.mean()
    daily_volatility = returns.std()
    sharpe_ratio = daily_returns / daily_volatility * np.sqrt(252)
    
    # Calculate the maximum drawdown
    drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()
    
    # Return the performance metrics
    performance_metrics = {
        "PNL": pnl,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": drawdown
    }
    return performance_metrics

def backtesting(returns, positions, start_index, end_index, capital, transaction_costs):
    # Select the returns data for the specified period
    returns = returns[start_index:end_index]
    
    # Calculate the cumulative returns
    cumulative_returns = (returns + 1).cumprod()
    
    # Calculate the portfolio value
    portfolio_value = capital / positions.sum() * (cumulative_returns * positions).sum(axis=1)
    
    # Calculate the transaction costs
    transaction_costs = (returns * transaction_costs).sum(axis=1).cumsum()
    
    # Deduct the transaction costs from the portfolio value
    portfolio_value = portfolio_value - transaction_costs
    
    # Calculate the performance metrics
    performance_metrics = calculate_performance_metrics(portfolio_value)
    
    # Return the portfolio value and performance metrics
    results = {
        "Portfolio Value": portfolio_value,
        "Performance Metrics": performance_metrics
    }
    return results
 
def plot_equity_curve(portfolio_value):
    # Convert the index of the portfolio value to a datetime object
    portfolio_value.index = pd.to_datetime(portfolio_value.index)

    # Calculate the drawdown
    drawdown = (portfolio_value / portfolio_value.cummax() - 1)

    # Calculate the maximum drawdown and the duration
    max_drawdown = drawdown.min()
    max_drawdown_start = drawdown.idxmin()
    max_drawdown_end = portfolio_value.loc[:drawdown.idxmin()].idxmax()
    max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days

    # Plot the equity curve
    fig, ax = plt.subplots()
    ax.plot(portfolio_value.index, portfolio_value, label="Equity")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Equity")
    ax.set_title("Equity Curve")

    # Add the maximum drawdown and the duration to the plot
    ax.annotate(f"Max Drawdown: {max_drawdown:.2%}", xy=(max_drawdown_start, portfolio_value[max_drawdown_start]), xycoords="data",
                xytext=(-150, 30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round", fc="w"))
    ax.annotate(f"Duration: {max_drawdown_duration} days", xy=(max_drawdown_start, portfolio_value[max_drawdown_start]), xycoords="data",
                xytext=(-150, -30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round", fc="w"))

    # Show the plot
    plt.legend()
    plt.show()

# Load the prices data into a pandas DataFrame
prices = pd.read_csv("prices.csv")

# Convert the prices data into returns
returns = calculate_returns(prices)

# Define the positions in the portfolio
positions = np.array([100, 200, 300, 400])

# Define the start and end indices of the backtesting period
start_index = 0
end_index = 1000

# Define the initial capital for the backtesting
capital = 100000

# Define the transaction costs as a percentage of the value of each trade
transaction_costs = 0.0005

# Apply the backtesting model
results = backtesting(returns, positions, start_index, end_index, capital, transaction_costs)

# Print the portfolio value and performance metrics
print("Portfolio Value:", results["Portfolio Value"])
print("Performance Metrics:", results["Performance Metrics"])

# Load the benchmark data into a pandas DataFrame
benchmark_prices = pd.read_csv("benchmark_prices.csv")

# Convert the benchmark prices data into returns
benchmark_returns = calculate_returns(benchmark_prices)

# Calculate the cumulative benchmark returns
cumulative_benchmark_returns = (benchmark_returns + 1).cumprod()

# Define the initial benchmark capital
benchmark_capital = 100000

# Calculate the benchmark portfolio value
benchmark_portfolio_value = benchmark_capital * cumulative_benchmark_returns

# Calculate the benchmark performance metrics
benchmark_performance_metrics = calculate_performance_metrics(benchmark_portfolio_value)

# Apply the backtesting model
results = backtesting(returns, positions, start_index, end_index, capital, transaction_costs)

# Compare the performance of the portfolio to the benchmark
excess_return = results["Portfolio Value"].iloc[-1] / benchmark_portfolio_value.iloc[-1] - 1
information_ratio = (results["Portfolio Value"].pct_change() - benchmark_portfolio_value.pct_change()).mean() / (results["Portfolio Value"].pct_change() - benchmark_portfolio_value.pct_change()).std()

# Print the portfolio value and performance metrics
print("Portfolio Value:", results["Portfolio Value"])
print("Performance Metrics:", results["Performance Metrics"])

# Print the benchmark performance metrics
print("Benchmark Performance Metrics:", benchmark_performance_metrics)

# Print the excess return and information ratio
print("Excess Return:", excess_return)
print("Information Ratio:", information_ratio)

# Plot the equity curve
plot_equity_curve(results["Portfolio Value"])
