import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,SimpleRNN
df = pd.read_csv("C:/Users/Swift/Downloads/Google_Stock_Price.csv", thousands = ',')
data = df['Open'].values.reshape(1,-1)
Scaler = MinMaxScaler(feature_range=(0,1))
data = pd.to_numeric(df['Open'],errors = 'coerce').dropna().values.reshape(-1,1)
data_scaled = Scaler.fit_transform(data)
train_size = int(len(data_scaled)*0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]
def create_dataset(data_set):
    x = []
    y = []
    for i in range(60,len(data_set)):
        x.append(data_set[i-60:i,0])
        y.append(data_set[i,0])

    return np.array(x),np.array(y)

x_train,y_train = create_dataset(train_data)
x_test,y_test = create_dataset(test_data)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
model = Sequential()
model.add(SimpleRNN(50,input_shape=(60,1),return_sequences = True))
model.add(SimpleRNN(50))
model.add(Dense(1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.summary()
model.fit(x_train,y_train,epochs = 20,batch_size = 32)
predicted = model.predict(x_test)
predicted = Scaler.inverse_transform(predicted)
real = Scaler.inverse_transform(y_test.reshape(-1,1))
plt.plot(real,color = 'blue',label = 'Actual Values')
plt.plot(predicted,color = 'red',label ='Predicted Values')
plt.title("Google Stock Price Prediction using RNN")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()