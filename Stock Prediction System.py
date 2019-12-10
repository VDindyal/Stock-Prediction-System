#Import numpy to make the arrays - the only allowed input to the neural networks as opposed to dataframes
import numpy as nmp
#Import matplotlib to visualize the results on charts
import matplotlib.pyplot as mpl
#Import pandas to import the datasets and manage them easily
import pandas as pnd

#Import the training set as data frames, select the correct columns, and conver to numpy array
training_data = pnd.read_csv('Apple_Train.csv')
#Create a dataframe with only the column 'Open'. Use .values to create numpy array.
set_train = training_data.iloc[:, 1:2].values

#Use normalization feature scaling
from sklearn.preprocessing import MinMaxScaler
#Create object from MinMaxScaler using a normalization feature scale (0,1)
mxobj = MinMaxScaler(feature_range = (0, 1))
#Create normalized training set by taking sc object and transforming it by fitting (gets minimum and max stock price) and scaling it (compute for each of the stock price the scaled stock price)
set_train_scaled = mxobj.fit_transform(set_train)
#All stock prices are now normalized between 0 and 1

# Create special data structure with 60 timesteps and 1 output. At each time t the RNN will look at the 60 stock prices before time t and time t. 
# Based on the trends it is capturing during these 60 time steps, it will predict the next outputs.
# 60 is the best number after much experimentation, 1 leads to overfitting, 20 is not enough to capture some trends, best is 60 previous financial days (3 months).
# The 1 output is at time t+1

# For each observation, prev_p train will contain the 60 previous stock prices
# next_p train will contain will contain the stock price the next financial day
prev_p = []
next_p = []

# We have to start at the first 60th financial day and go to the last day of our observation
# For prev_p append each time from the last 60 days (it's to 59, since upper bound excluded), with column 0
# For next_p append at time t+1 which is 60. Thus it is i.

for i in range(60, 1258):
    prev_p.append(set_train_scaled[i-60:i, 0])
    next_p.append(set_train_scaled[i, 0])
    
# prev_p and next_p are lists, we need to make them into arrays using numpy
prev_p, next_p = nmp.array(prev_p), nmp.array(next_p)

# Add new dimension to the data structure to add more indictaors tfor upward and downward trends
# Use reshape function to add another dimension to array
# Input shape should be a 3D array for Keras to use, containing the batch size (total number of stock prices from 2012 to 2016), the timesteps (60), and the new indicators (closing price)
prev_p = nmp.reshape(prev_p, (prev_p.shape[0], prev_p.shape[1], 1))

#Import the Keras Sequential class to create a neural network object representing a sequence of layers
from keras.models import Sequential
#Import the Keras Dense class to add the output layer
from keras.layers import Dense
#Import the LSTM class to add the LSTM layers
from keras.layers import LSTM
#Import the Dropout class to add some dropout regularisation
from keras.layers import Dropout

# Use the sequential class to introduce regressor as sequence of layers
reg = Sequential()

#Adding the first LSTM layer and Dropout regularisation of 20% to avoid overfitting
reg.add(LSTM(units = 50, return_sequences = True, input_shape = (prev_p.shape[1], 1)))
reg.add(Dropout(0.2))

#Adding the second LSTM layer and Dropout regularisation of 20% to avoid overfitting
#No need to specify input shape anymore. It is recognized automatically from the units = 50 in the first LSTM layer
reg.add(LSTM(units = 50, return_sequences = True))
reg.add(Dropout(0.2))

#Adding the third LSTM layer and Dropout regularisation of 20% to avoid overfitting
reg.add(LSTM(units = 50, return_sequences = True))
reg.add(Dropout(0.2))

#Adding the fourth LSTM layer and Dropout regularisation of 20% to avoid overfitting
#This is the last LSTM layer, and thus we do not need to return anymore sequences
reg.add(LSTM(units = 50))
reg.add(Dropout(0.2))

#Adding the output layer with stock price t+1
reg.add(Dense(units = 1))

#Compiling the RNN with an optimizer
reg.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fit RNN to training set. Use epochs to forward and backward propogate. 
#There wasn't a convergence of the loss with 25 or 50 epochs, 100 is optimal.
reg.fit(prev_p, next_p, epochs = 100, batch_size = 32)

#Import the actual stock prices to compare against
testing_data = pnd.read_csv('Apple_Test.csv')
actual_price = testing_data.iloc[:, 1:2].values

#Gather the correct inputs and perform scaling for the predicted stock prices
total_data = pnd.concat((training_data['Open'], testing_data['Open']), axis = 0)
inputs = total_data[len(total_data) - len(testing_data) - 60:]
inputs = inputs.reshape(-1,1)
inputs = mxobj.transform(inputs)

predict_p = []
for i in range(60, 80):
    predict_p.append(inputs[i-60:i, 0])
predict_p = nmp.array(predict_p)
predict_p = nmp.reshape(predict_p, (predict_p.shape[0], predict_p.shape[1], 1))
predicted_price = reg.predict(predict_p)
predicted_price = mxobj.inverse_transform(predicted_price)

#Graphing the result
mpl.plot(actual_price, color = 'green', label = 'Real Price')
mpl.plot(predicted_price, color = 'red', label = 'Predicted Price')

mpl.title('Apple Stock Prediction')
mpl.xlabel('Time')
mpl.ylabel('Stock Price')
mpl.legend()
mpl.show()