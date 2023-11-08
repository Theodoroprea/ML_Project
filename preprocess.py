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
        
        # HEAT MAP
        # heat_map(df)
        
        # From the heatmap, we can see that the following have quite a bit of corrolation with avg_rating:
        # - age (for what age group is the game made)
        # - weight (weight of game)
        # - owned (how many games are owned in the world)
        # - avg_time (average time of games)
        # - num_votes (how many number of votes a game has)
        
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
        
        
    if(pp_type == "img"):
        df.drop(['rank', 'bbg_url', 'geek_rating', 'game_id', 'mechanic', 'category', 'designer'], axis=1,inplace=True) # Drop the ones we do not need
        
        
def heat_map(df):
    plt.figure(figsize = (10,10)) # Adjusting figure size.
    sns.heatmap(df.corr(), annot=True)  # 'annot' displays values in cells, 'cmap' sets the color map
    plt.title("Correlation Heatmap")  # Add a title to your heatmap
    plt.show()  # Display the heatmap
    
def pair_plot(df):
    sns.set(font_scale = 1.5)
    sns.pairplot(df[['age','weight','owned','avg_time','num_votes', 'avg_rating']],height = 2.2, aspect = 0.9)
    plt.show()  # Display the heatmap