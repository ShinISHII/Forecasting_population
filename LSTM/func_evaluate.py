def calculate_mape(true_values, predicted_values):
    true_values, predicted_values = np.array(true_values), np.array(predicted_values)
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100


# MSEを計算する関数
def calculate_mse(true_values, predicted_values):
    true_values, predicted_values = np.array(true_values), np.array(predicted_values)
    return np.mean((true_values - predicted_values) ** 2)

def calculate_rmse(true_values, predicted_values):
    return (calculate_mse(true_values, predicted_values) ** 0.5)