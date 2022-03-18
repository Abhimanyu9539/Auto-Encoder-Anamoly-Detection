import numpy as np

def init(test_data, model, columns):
    columns = list(columns)
    columns.remove("Class")
    test_data = test_data[columns]
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]

    x_test = test_data.values
    predict = model.predict(x_test)

    diff_array = np.subtract(x_test, predict)
    squared_array = np.square(diff_array)
    mse = squared_array.mean()
    return mse