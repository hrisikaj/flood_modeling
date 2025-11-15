from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
print("End of Feature engineering")


##########################################################
# building xgboost model

X = train_data.drop(['FloodProbability', 'id'], axis=1)
y = train_data['FloodProbability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
model = XGBRegressor(objective='reg:squarederror', tree_method='hist', predictor='gpu_predictor') # base xgb model

# parameter grid for xgb
param_grid_xgb = {
    'n_estimators': [400, 600],
    'max_depth': [3,6],
    'learning_rate': [0.7, 1],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.8, 0.9]
}
grid_search_xgb = GridSearchCV(estimator=model, param_grid=param_grid_xgb, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

# finding best model
best_model = grid_search_xgb.best_estimator_
y_pred = best_model.predict(X_test)
print("Best parameters:", grid_search_xgb.best_params_)
print("Best CV R² score:", grid_search_xgb.best_score_)
joblib.dump(best_model, 'best_sgd_model.joblib')

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² score: {r2}")
print(f"Mean Squared Error: {mse}")