import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("./nairobi-office-price-dataset.csv")
x = data['SIZE'].values
y = data['PRICE'].values

# Initialize parameters
m = np.random.rand()
c = np.random.rand()
learning_rate = 0.01
epochs = 80


def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost


# Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * x + c
    d_m = (-2 / N) * sum(x * (y - y_pred))
    d_c = (-2 / N) * sum(y - y_pred)
    m -= learning_rate * d_m
    c -= learning_rate * d_c
    return m, c

# Training loop
for epoch in range(epochs):
    m, c = gradient_descent(x, y, m, c, learning_rate)
    mse = mean_squared_error(y, m * x + c)
    print(f"Epoch {epoch+1} || MSE: {mse} || Weight: {m} || Bias: {c}")

# Plot the line of best fit
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m * x + c, color='red', label='Best Fit Line')
plt.xlabel("Office Size")
plt.ylabel("Office Price")
plt.legend()
plt.show()

# Prediction
office_size = float(100)
print(m, c)
predicted_price =(m * office_size) + c
print(f"Predicted price for a 100 sq. ft. office: {predicted_price}")