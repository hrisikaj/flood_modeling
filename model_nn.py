import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import joblib


train_nn = pd.read_csv('/home/neera/university/floodprediction/data/train.csv')
test_nn = pd.read_csv('/home/neera/university/floodprediction/data/test.csv')

# neural network model
# split training data into train and test for this model
X = train_nn.drop(['FloodProbability', 'id'], axis=1)
y = train_nn['FloodProbability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# nerual network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Make predictions on X_test
y_pred = model.predict(X_test)

# Compute R²
r2 = r2_score(y_test, y_pred)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² score:", r2)
print("RMSE:", rmse)

joblib.dump(model, 'keras_nn_model.pkl')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1) 
fig1 = plt.figure()
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig("Train_Val_Loss.png")
plt.close(fig1)

accuracy = history.history['mae']
val_acc = history.history['val_mae']
epochs = range(1, len(accuracy) + 1)
fig2 = plt.figure()
plt.plot(epochs, accuracy, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
# plt.show()
plt.savefig("Train_val_acc.png") 
plt.close(fig2)

