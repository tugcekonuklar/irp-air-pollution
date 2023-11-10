
import model_base as mb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#
# def tune_and_evaluate(df):
#     n_iter_search = 10
#     random_state = 42
#
#     # Define your features and target variable
#     train_data, validation_data, test_data = mb.split_data(df)
#     # Extract the features
#     X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
#     # Extract the target variable
#     y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
#
#     print(f'Started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
#
#     param_distributions = {
#         'n_estimators': sp_randint(100, 1000),
#         'max_depth': sp_randint(3, 30),
#         'min_samples_split': sp_randint(2, 11),
#         'min_samples_leaf': sp_randint(1, 11),
#         'max_features': ['auto', 'sqrt', 'log2', None]
#     }
#
#     # RandomizedSearchCV
#     model = RandomForestRegressor()
#
#     # Create the TimeSeriesSplit object
#     tscv = TimeSeriesSplit(n_splits=5)
#
#     # Create the RandomizedSearchCV object
#     random_search = RandomizedSearchCV(model, param_distributions=param_distributions,
#                                        n_iter=n_iter_search, scoring='neg_mean_squared_error',
#                                        cv=tscv, n_jobs=1, random_state=random_state, verbose=1)
#
#     print(f'Fitted {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
#     # Fit the model on the training data
#     try:
#         random_search.fit(X_train, y_train)
#     except Exception as ex:
#         print(ex)
#
#     # Print the best parameters and best score from the training
#     print(f"Best parameters for Random Forest : {random_search.best_params_}")
#     print(f"Best score Random Fores : {-random_search.best_score_}")
#
#     print(f'Prediction started  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
#
#     # Check the performance on the validation set
#     y_val_pred = random_search.predict(X_val)
#     mb.evolve_error_metrics(y_val, y_val_pred)
#     mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)
#
#     # Use the best estimator to predict on the test set
#     y_test_pred = random_search.best_estimator_.predict(X_test)
#     mb.evolve_error_metrics(y_test, y_test_pred)
#     mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)
#
#     print(f'Finished {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
#     # Return the best estimator and the MSE scores
#     return random_search.best_estimator_

#
# def train_and_evolve(df):
#     train_data, validation_data, test_data = mb.split_data(df)
#     # Extract the features
#     # Normalize the features
#     X_train, X_val, X_test =  mb.scale_features(train_data, validation_data, test_data, MinMaxScaler())
#     # Extract the target variable
#     y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
#
#     # ANN model architecture
#     model = Sequential()
#     model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#
#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#
#     # Train the model
#     model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)
#
#     # Evaluate the model
#     loss = model.evaluate(X_test, y_test, verbose=0)
#     print(f'Model Loss on Test Data: {loss}')
#
#     # VALIDATION Prediction and Evolution
#     y_val_pred = model.predict(X_val)
#
#     print(y_val_pred)
#
#     # Validation Error Metric
#     mb.evolve_error_metrics(y_val, y_val_pred)
#     mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)
#
#     # TEST Prediction and Evolution
#     y_test_pred = model.predict(X_test)
#
#     print(y_test_pred)
#
#     # Test Error Metric
#     mb.evolve_error_metrics(y_test, y_test_pred)
#     mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)
#
#     # Plot
#     mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
#     mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
