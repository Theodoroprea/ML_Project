import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        plt.figure(figsize = (10,10)) # Adjusting figure size.
        sns.heatmap(df.corr(), annot=True)  # 'annot' displays values in cells, 'cmap' sets the color map
        plt.title("Correlation Heatmap")  # Add a title to your heatmap
        plt.show()  # Display the heatmap
        
    if(pp_type == "img"):
        df.drop(['rank', 'bbg_url', 'geek_rating', 'game_id', 'mechanic', 'category', 'designer'], axis=1,inplace=True) # Drop the ones we do not need