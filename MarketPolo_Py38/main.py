# Description : This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM)
#               to predict the closing stock price of a corporation (Apple Inc.) using the past 60 day stock price.

# Import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Constants
SCALER = MinMaxScaler(feature_range=(0, 1))


def get_and_check_stock():
    stock = input("Enter a ticker symbol: ")
    print("Checking the stock " + stock + " ...")

    try:
        quote = web.DataReader(stock, data_source='yahoo', start='2021-01-04', end='2021-01-04')
    except:
        print("Could not find information on the ticker: " + stock)
        return None

    return stock


def get_data(stock):
    # Get the stock quote
    df = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end='2019-12-17')

    # Show the data
    pd.set_option('display.max_columns', None)
    print(df)

    # Get the number of rows and columns in the data set
    # print(df.shape)

    # Visualize the closing price history
    # plt.figure(figsize=(16,8))
    # plt.title('Closing Price History')
    # plt.plot(df['Close'])
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Close Price USD ($)', fontsize=18)
    # plt.show()

    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values  # add np

    # Get the number of rows to train the model on
    # using 80% of the data for training and 20% for testing
    training_data_len = math.ceil(len(dataset) * .8)
    print("Length of training dataset: " + str(training_data_len))

    # Scale the data
    scaled_data = SCALER.fit_transform(dataset)
    print(scaled_data)

    return data, dataset, scaled_data, training_data_len


def preprocessing(scaled_data, training_data_len):

    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    # Split the data in x_train and y_train
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:  # use 60 and 61 to see differences in iterations
            print(x_train)
            print(y_train)

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    # samples, #timesteps, #features --LSTM requireing 3D data with these params
    # x_train only 2D so reshape it
    # in this example #features is only 1 which is the Closing price
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def build_and_test_lstm(x_train, y_train, dataset, scaled_data, training_data_len):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # 50 is number of neurons
    model.add(LSTM(50, return_sequences=False))
    # another LSTM layer but return is False because wont add anymore
    model.add(Dense(25))
    # another densly connected NN layer with 25 neurons
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    # Create a new array containing scaled values from 1543 to 2003 (remaining 20%)
    test_data = scaled_data[training_data_len - 60:, :]
    # Create teh data sets x_tests, y_tests
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = SCALER.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(rmse)
    # lower number means more accurate prediction

    return predictions, model


def visualization(data, training_data_len, predictions):
    # Plot the data
    plt.style.use('fivethirtyeight')
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
    plt.show()

    # Show the valid and predicted prices
    print(valid)


def make_prediction(model, stock):
    # Get the quote
    apple_quote = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end='2019-12-17')
    # Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    # Get the last 60 day closing price values and convert the dataframe to an array
    last_60_days = new_df[-60:].values
    # Scale the data to be values between 0 and 1
    last_60_days_scaled = SCALER.fit_transform(last_60_days)
    # Create an empty list
    x_test = []
    # Append the past 60 days
    x_test.append(last_60_days_scaled)
    # Convert the X_test set to numpy array
    x_test = np.array(x_test)
    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # Get the predicted scale price
    predict_price = model.predict(x_test)
    # undo the scaling
    predict_price = SCALER.inverse_transform(predict_price)
    print(predict_price)

    # Get actual quote of next day
    apple_quote2 = web.DataReader(stock, data_source='yahoo', start='2019-12-18', end='2019-12-18')
    print(apple_quote2['Close'])


def main():
    print("Hagimemasho! \n")

    ticker = get_and_check_stock()

    if ticker is not None:
        close_price_data, close_dataset, scaled_close_dataset, training_length = get_data(ticker)

        xtrain, ytrain = preprocessing(scaled_close_dataset, training_length)

        tested_predictions, model = build_and_test_lstm(xtrain,
                                                        ytrain,
                                                        close_dataset,
                                                        scaled_close_dataset,
                                                        training_length)

        visualization(close_price_data, training_length, tested_predictions)

        make_prediction(model, ticker)
    else:
        print("Invalid ticker. Please check symbol and try again.")


if __name__ == "__main__":
    main()
