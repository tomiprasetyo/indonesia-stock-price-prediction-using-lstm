# LSTM Hyperparameter Tuning for Stock Price Prediction

This Python script demonstrates the process of training an LSTM (Long Short-Term Memory) model for stock price prediction and optimizing its hyperparameters using Optuna.

## Overview

The script performs the following steps:

1. **Data Collection**: Reads stock price data from a CSV file.
2. **Data Preprocessing**: Handles missing values, normalizes the data, and splits it into training and testing sets.
3. **Model Development**: Defines the LSTM model architecture and functions for training and validation.
4. **Hyperparameter Tuning**: Uses Optuna to find the optimal hyperparameters for the LSTM model, including the hidden size, number of stacked layers, learning rate, and batch size.
5. **Model Training**: Trains the LSTM model with the optimal hyperparameters and evaluates its performance on the training and testing sets.
6. **Model Evaluation**: Calculates various evaluation metrics, such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), and R-squared (R²).
7. **Future Prediction**: Generates future stock price predictions for 7, 14, and 30 days ahead.
8. **Visualization**: Provides visualizations of the actual and predicted stock prices throughout the process.
9. **Model Saving**: Saves the trained LSTM model to a file for future use.

## Requirements

The script requires the following Python libraries:

- `pandas`
- `numpy`
- `plotly`
- `torch`
- `sklearn`
- `optuna`

You can install these dependencies using pip:

Copy

`pip install pandas numpy plotly torch scikit-learn optuna`

## Usage

1. Place the stock price data in a CSV file named `AKRA.JK.csv` in the same directory as the Python script.
2. Run the Python script using your preferred method (e.g., `python lstm-hyperparameter-tuning.py`).
3. The script will automatically perform the hyperparameter tuning, model training, and evaluation, and display the results as well as the predicted stock prices.
4. The trained model will be saved to a file named `model.pth` in the same directory.

## Customization

You can customize the script by modifying the following parameters:

- `lookback`: The number of previous days to consider as input features.
- `num_epochs`: The number of training epochs.

Additionally, you can adjust the hyperparameter search space in the `objective` function to explore different ranges for the hidden size, number of stacked layers, learning rate, and batch size.

## License

This project is licensed under the [MIT License](LICENSE).
