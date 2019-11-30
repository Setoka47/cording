import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(0)

model = Sequential()
model.add(Dense(input_dim=2, units=1))  # 2入力、ユニット数1
model.add(Activation('sigmoid'))  # シグモイド関数を使用
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

# データ
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [0], [0], [1]])

# 学習
model.fit(X, T, epochs=20, batch_size=1)

# 実行
Y = model.predict_classes(X, batch_size=1)
print()
print("TEST")
print(Y == T)
