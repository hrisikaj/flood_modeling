import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor



train_data = pd.read_csv('/home/neera/university/floodprediction/data/train.csv')
test_data = pd.read_csv('/home/neera/university/floodprediction/data/test.csv')

print("\n2. FEATURE ENGINEERING")
print("="*50)

# Combine the two features and normalize
train_data['CombinedPreparednessPlanning'] = train_data[['InadequatePlanning', 'IneffectiveDisasterPreparedness']].mean(axis=1)

# Normalize the new feature to the range of 1 to 17
scaler = MinMaxScaler(feature_range=(1, 17))
train_data['CombinedPreparednessPlanning_Scaled'] = scaler.fit_transform(train_data[['CombinedPreparednessPlanning']])

# Drop the original features and the intermediate combined feature
train_data = train_data.drop(['InadequatePlanning', 'IneffectiveDisasterPreparedness'], axis=1)

# Create an interaction term between Urbanization and PopulationScore
train_data['Urbanization_Population_Interaction'] = train_data['Urbanization'] * train_data['PopulationScore']

# Scale the interaction term to the range of 1 to 17
scaler = MinMaxScaler(feature_range=(1, 17))
train_data['Urbanization_Population_Interaction_Scaled'] = scaler.fit_transform(train_data[['Urbanization_Population_Interaction']])
# Bin the Urbanization_Population_Interaction_Scaled feature into 17 bins
train_data['Urbanization_Population_Interaction_Binned'] = pd.cut(train_data['Urbanization_Population_Interaction_Scaled'], bins=17, labels=False, include_lowest=True)

train_data = train_data.drop(['Urbanization_Population_Interaction', 'CombinedPreparednessPlanning', 'Urbanization_Population_Interaction_Scaled'], axis=1)

print("Finished feature engineering")

X = train_data.drop(['FloodProbability', 'id'], axis=1)
y = train_data['FloodProbability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


# def build_nnmodel():
#     model = Sequential()
#     model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#     return model

# nn_regressor = KerasRegressor(
#     model=build_nnmodel,
#     epochs=5,
#     batch_size=16,
#     verbose=0
# )

from sklearn.neural_network import MLPRegressor

nn_regressor = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=100)

models_l0 = [
    ('sgd', make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, penalty="l2", alpha = 0.0001, learning_rate="adaptive"))),
    ('nn', nn_regressor)
]

meta_model = LinearRegression()
# Stacking regressor
stacking_regressor = StackingRegressor(estimators=models_l0, final_estimator=meta_model)

stacking_regressor.fit(X_train, y_train.values.ravel())

# Predict
y_pred = stacking_regressor.predict(X_test)

# Evaluate
print("R2 score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))