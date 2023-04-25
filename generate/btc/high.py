import pandas as pd
import datetime as dt
import numpy as np

from csv import writer

from tensorflow.keras.models import Sequential
from keras.layers import GRU, Dense, Dropout,Bidirectional
from keras.models import load_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('./data/clean/btc_high.csv')

df['H-L'] = df['high'] - df['low']
MA_1 = 7
MA_2 = 14
MA_3 = 21

df[f'SMA_{MA_1}'] = df['high'].rolling(window=MA_1).mean()
df[f'SMA_{MA_2}'] = df['high'].rolling(window=MA_2).mean()
df[f'SMA_{MA_3}'] = df['high'].rolling(window=MA_3).mean()

df[f'SD_{MA_1}'] = df['high'].rolling(window=MA_1).std()
df[f'SD_{MA_3}'] = df['high'].rolling(window=MA_3).std()
df.dropna(inplace=True)

pre_day =7
scala_x = MinMaxScaler(feature_range=(0,1))
scala_y = MinMaxScaler(feature_range=(0,1))
cols_x = ['low', 'open', 'close', 'O-C', f'SMA_{MA_1}_high', f'SMA_{MA_2}_high', f'SMA_{MA_3}_high', f'SD_{MA_1}_high',
               f'SD_{MA_3}_high']
cols_y = ['high']
scaled_data_x = scala_x.fit_transform(df[cols_x].values.reshape(-1, len(cols_x)))
scaled_data_y = scala_y.fit_transform(df[cols_y].values.reshape(-1, len(cols_y)))

x_total = []
y_total = []

for i in range(pre_day, len(df)):
    x_total.append(scaled_data_x[i-pre_day:i])
    y_total.append(scaled_data_y[i])

    # TEST SIZE
test_size = (int)(len(scaled_data_y) * 0.2)
print(test_size)

x_train = np.array(x_total[:len(x_total)-test_size])
x_test = np.array(x_total[len(x_total)-test_size:])
y_train = np.array(y_total[:len(y_total)-test_size])
y_test = np.array(y_total[len(y_total)-test_size:])

# BUILD MODEL
model = Sequential()

model.add(GRU(units=60, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.1))
model.add(GRU(units=64, return_sequences=True, input_shape=(1, len(cols_x))))
model.add(Dropout(0.1))
model.add(Bidirectional(GRU(units=8, return_sequences=False)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=120, steps_per_epoch=40, use_multiprocessing=True,validation_data=(x_test, y_test))

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

predict_price = model.predict(x_test)

predict_high_value = model.predict(x_train)
predict_high_value = scala_y.inverse_transform(predict_high_value)
y_train_high_value = scala_y.inverse_transform(y_train)


error = pd.concat([pd.DataFrame(predict_high_value,columns=['predict_high_value'],index=None), pd.DataFrame(y_train_high_value,columns=['y_train_high_value'],index=None)], axis=1)
error['error_high_value'] = error['predict_high_value'] - error['y_train_high_value']
error.to_csv('./data/predict/result_high_value.csv',encoding = 'utf-8-sig')