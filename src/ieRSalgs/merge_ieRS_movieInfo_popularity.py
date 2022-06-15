# merge the ieRS_item_popularity.csv with the ieRS_movieInfo_emotions_g20.csv
import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
# There will be SettingWithCopyWarning here, use the above code to hide the warnings
  
import pandas as pd
import numpy as np
import csv

filepath = './data/ieRS_item_popularity.csv'
ieRS_item_popularity = pd.read_csv(filepath)
    # ['item', 'count', 'rank']
ieRS_item_popularity = ieRS_item_popularity.rename(columns = {'item': 'movie_id', 'count': 'ieRS_count', 'rank': 'ieRS_rank' })    
print(ieRS_item_popularity.shape)    
# (9064, 3)
    
    
filepath = './data/ieRS_movieInfo_emotions_g20.csv' 
ieRS_movieInfo_emotions_g20 = pd.read_csv(filepath)
print(ieRS_movieInfo_emotions_g20.shape)
# (9064, 21) 
print(ieRS_movieInfo_emotions_g20.columns)
    # ['movie_id', 'imdb_id', 'title(year)', 'title', 'year', 'runtime', 'genre', 'aveRating', 'director', 'writer', 'description', 'cast', 'poster', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

forDB = pd.merge(ieRS_movieInfo_emotions_g20, ieRS_item_popularity, how='inner', on=['movie_id'])
print(forDB.columns)

forDB.to_csv('./data/ieRS_movieInfo_emotions_ranking.csv', index = False)