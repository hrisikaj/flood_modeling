import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

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



# Fine tuning the sgd model
X = train_data.drop(['FloodProbability', 'id'], axis=1)
y = train_data['FloodProbability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
pipeline = Pipeline([('scaler', StandardScaler()), ('sgd', SGDRegressor(random_state=50))])

# parameter grid for fine tuning
param_grid_sgd = {
    'sgd__loss': ['squared_error', 'huber'],
    'sgd__penalty': ['l1', 'l2'],
    'sgd__alpha': [0.0001, 0.001],
    'sgd__learning_rate': ['optimal', 'adaptive'],
    'sgd__max_iter': [1000, 1500]

}

grid_search_sgd = GridSearchCV(pipeline, param_grid_sgd, cv=5, scoring='r2', n_jobs=-1)
grid_search_sgd.fit(X_train, y_train)
print("Best parameters:", grid_search_sgd.best_params_)
print("Best CV R² score:", grid_search_sgd.best_score_)

# Evaluate on test data
best_model = grid_search_sgd.best_estimator_
joblib.dump(best_model, 'best_sgd_model.joblib')
print("model saved!")
test_r2 = best_model.score(X_test, y_test)
print("Test R² score:", test_r2)
y_pred = best_model.predict(X_test)

print("final metrics: \n")
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² score: {r2}")
print(f"Mean Squared Error: {mse}")
