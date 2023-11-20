import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings


def data_preprocess(pp_type):
    df = pd.read_csv('./data/bgg_db_1806.csv')
    if(pp_type == "quant"):
        df.drop(['names','rank', 'bgg_url', 'geek_rating', 'game_id', 'mechanic', 'category', 'designer', 'image_url'], axis=1,inplace=True) # Drop the columns we do not need
        
        df.dropna(axis = 0,inplace = True) # Any columns that have missing informating need to be dropped.
        
        df.drop(df[df['num_votes'] == 0].index,inplace = True) # If there are any games that have 0 votes, then we should drop them
        
        # If min_players > max_players, then we switch the 2.
        a = (df['min_players'] > df['max_players'])
        df.loc[a,['min_players','max_players']] = df.loc[a, ['max_players','min_players']].values
        
        # If min_time > max_time, then we switch the 2.
        b = (df['min_time'] > df['max_time'])
        df.loc[b,['min_time','max_time']] = df.loc[b,['max_time','min_time']].values
        
        # HEAT MAP
        heat_map(df)
        
        # From the heatmap, we can see that the following have quite a bit of corrolation with avg_rating:
        # - age (for what age group is the game made)
        # - weight (weight of game)
        # - owned (how many games are owned in the world) (Remove)
        # - avg_time (average time of games)
        # - num_votes (how many number of votes a game has) (Remove)
        
        # PAIRPLOT
        # pair_plot(df)
        
        # owned copies vs rating
        # df['new_num_votes'] = df['num_votes'].apply(lambda x: 1 if x>df['num_votes'].mean() else 0)
        # sns.scatterplot(x='avg_rating',y='owned',data=df,hue = 'new_num_votes',legend = 'full')
        # plt.show()
        
        # num_votes vs rating 
        # df['new_num_votes'] = df['num_votes'].apply(lambda x: 1 if x>5000 else 0)
        # sns.scatterplot(x='avg_rating',y='num_votes',data=df,hue = 'new_num_votes',legend = 'full')
        # plt.show()
        
        # weight vs rating
        # df['new_num_votes'] = df['num_votes'].apply(lambda x: 1 if x>1000 else 0)
        # sns.scatterplot(x='avg_rating',y='weight',data=df,hue = 'new_num_votes',legend = 'full')
        # plt.show()
        
        X = df.drop(['num_votes', 'owned','avg_rating'], axis=1) # Everything but the avg_rating
        y = df['avg_rating'] # Labels are the avg_ratings
        
        print(X.head())
        # Transform into numpy arrays
        X = X.values
        y = y.values
        
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42)
        
        alg_choice = int(input("Would you like to run (1) random forest or (2) MLP?\n"))
        
        #### MOVE THIS TO DIFF FUNCTION
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
                # Use root mean squared error instead of MSE. (MRSE standard for reg)
                grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                
                best_mlp_model = grid_search.best_estimator_
                print(best_mlp_model.coefs_)
                y_test_prediction = best_mlp_model.predict(X_test)
            else:
                raise Exception("Please use either 1 or 2 to answer.")
            
            mse = mean_squared_error(y_test, y_test_prediction)
            print(f"The mean squared error is: {mse}")
        
    if(pp_type == "img"):
        df.drop(['rank', 'bbg_url', 'geek_rating', 'game_id', 'mechanic', 'category', 'designer'], axis=1,inplace=True) # Drop the ones we do not need
        
# Confusion matrix
def heat_map(df):
    plt.figure(figsize = (10,10)) # Adjusting figure size.
    sns.heatmap(df.corr(), annot=True)  # 'annot' displays values in cells, 'cmap' sets the color map
    plt.title("Correlation Heatmap")  # Add a title to your heatmap
    plt.show()  # Display the heatmap
    
def pair_plot(df):
    sns.set(font_scale = 1.5)
    sns.pairplot(df[['age','weight','owned','avg_time','num_votes', 'avg_rating']],height = 2.2, aspect = 0.9)
    plt.show()  # Display the heatmap