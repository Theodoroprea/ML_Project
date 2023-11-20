import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor

def quant_models(X_train, X_test, y_train, y_test):
    alg_choice = int(input("Would you like to run (1) random forest or (2) MLP?\n"))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if alg_choice == 1:
            param_grid = {
                'n_estimators': [50, 100, 200, 300], # number of trees in forest
                'max_depth': [None, 10, 30, 60, 90]
            }
            # Using random forest regressor
            rf = RandomForestRegressor(random_state=42)
            
            # 5 Fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Grid search to get best combo
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Get the best model and run the prediction on test data
            
            best_rf_model = grid_search.best_estimator_
            y_test_prediction = best_rf_model.predict(X_test)    
        elif alg_choice == 2:
            # MLP
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (50,50,50)],
                'solver': ['lbfgs', 'adam'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                'max_iter': [500, 1000, 2000, 5000]
            }
            
            # Setup the regressor
            mlp = MLPRegressor(random_state=42)
            # 5 Fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Grid search to get best combo (n_jobs => Number of jobs to run in parallel, -1 means using all processors)
            grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_mlp_model = grid_search.best_estimator_
            y_test_prediction = best_mlp_model.predict(X_test)
        else:
            raise Exception("Please use either 1 or 2 to answer.")
        
        rmse = mean_squared_error(y_test, y_test_prediction, squared=False)
        print(f"The root mean squared error is: {rmse}")
        best_params = grid_search.best_params_
        print(f"The best parameters being: {best_params}")