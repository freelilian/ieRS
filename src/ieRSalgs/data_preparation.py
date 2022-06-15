import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
# There will be SettingWithCopyWarning here, use the above code to hide the warnings
  
import pandas as pd
import numpy as np
import csv

## import the raw data:  
filepath = '../data_v7/moviesnew.csv'
emotionsfinalcollection = pd.read_csv(filepath)
# print(emotionsfinalcollection.columns)

# eRS_movie_info = emotionsfinalcollection[['movielensId', 'titleId', 'name', 'release_year', 'movie_genres', 'movie_rating', 'movie_directors', 'movie_writers', 'plot', 'movie_stars', 'poster']]
# eRS_movie_info = eRS_movie_info.rename(columns= {'movielensId': 'movie_id', 'titleId': 'imdb_id', 'name': 'title', 'release_year': 'year', 'movie_genres': 'genre', 'movie_rating': 'aveRating', 'movie_directors': 'director', 'movie_writers': 'writer', 'plot': 'description', 'movie_stars': 'cast', 'poster'})

emotions = emotionsfinalcollection[['movielensId', 'titleId', 'signature.anger', 'signature.anticipation', 'signature.disgust', 'signature.fear', 'signature.joy', 'signature.sadness', 'signature.surprise', 'signature.trust']]
emotions['titleId'] = np.int64(emotions['titleId'].str[2:])
print(emotions.shape)
# (9221, 10)
unique_movielensId = emotions.movielensId.unique()
print(len(unique_movielensId))
# 9221, good


## check if all the movies are existed in rssa_movie_info.csv
## 
filepath = './data/rssa_movie_info.csv'
movie_info = pd.read_csv(filepath)
print(movie_info.shape)
# ['movie_id', 'imdb_id', 'title(year)', 'title', 'year', 'runtime', 'genre', 'aveRating', 'director', 'writer', 'description', 'cast', 'poster', 'count', 'rank']
# (57533, 15)
# print(movie_info.dtypes)
movie_info = movie_info[['movie_id', 'imdb_id', 'title(year)', 'title', 'year', 'runtime', 'genre', 'aveRating', 'director', 'writer', 'description', 'cast', 'poster']]
    # drop count and ranking
emotions = emotions.rename(columns = {'movielensId' : 'movie_id', 'titleId' : 'imdb_id', 'signature.anger': 'anger', 'signature.anticipation': 'anticipation', 'signature.disgust': 'disgust', 'signature.fear': 'fear', 'signature.joy': 'joy', 'signature.sadness': 'sadness', 'signature.surprise': 'surprise', 'signature.trust': 'trust'})
emotions_copy = emotions.drop(columns=['imdb_id'])
    # ['movie_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
print(emotions_copy.columns)
intersection = pd.merge(movie_info, emotions_copy, how='inner', on=['movie_id'])
print(intersection.shape)
# (9069, 22)
# 9069 common movies, reason: I lost some movies when download the movie info.
# Decision: use movies in the intersection


movielensID = intersection.movie_id.unique()
print(len(movielensID))
# 9069
## extract ratings
filepath = './data/ml-25m_ratings_ready.csv'
ml_25m_ratings = pd.read_csv(filepath)
    # ['userId', 'movieId', 'rating', 'timestamp']
    # (24702320, 4)
ml_25m_ratings = ml_25m_ratings.rename(columns = {'userId' : 'user_id', 'movieId' : 'movie_id'})
print(ml_25m_ratings.shape)
# (24702320, 4), that is how the dataset name 'ml-25m' coming from

ieRS_ratings_noRatingCountThreshold = ml_25m_ratings[ml_25m_ratings['movie_id'].isin(movielensID)]
print(ieRS_ratings_noRatingCountThreshold.shape)
    # (15443463, 4)
movid_id_g0 = ieRS_ratings_noRatingCountThreshold.movie_id.unique() 
print(len(movid_id_g0))
# 9069
ieRS_movieInfo_emotions_g0 = intersection[intersection['movie_id'].isin(movid_id_g0)] 
# 9609 movies
ieRS_movieInfo_emotions_g0.to_csv('./data/ieRS_movieInfo_emotions_g0.csv', index= False)

ieRS_emotions_g0 = emotions[emotions['movie_id'].isin(movid_id_g0)] 
# 9609 movies
ieRS_emotions_g0.to_csv('./data/ieRS_emotions_g0.csv', index= False)
###########################################################################################################
################## make sure each user in the offline dataset rated at least 20 movies#####################
###########################################################################################################
users, counts = np.unique(ieRS_ratings_noRatingCountThreshold['user_id'], return_counts = True)
users_rating_count = pd.DataFrame({'user_id': users, 'count': counts}, columns = ['user_id', 'count'])
users_rating_count_sorted = users_rating_count.sort_values(by = 'count', ascending = True)
# print(users_rating_count_sorted.head(10))   
#    min count is 1, some users only rate 1 movies
ieRS_ratings_noRatingCountThreshold.to_csv('./data/eRS_ratings_g0.csv', index = False) 


users_rating_count_g20 = users_rating_count[users_rating_count['count'] >= 20]
# print(users_rating_count_g20.shape)
    
user_id_g20 = users_rating_count_g20.user_id.unique()
ieRS_ratings_g20 = ieRS_ratings_noRatingCountThreshold[ieRS_ratings_noRatingCountThreshold['user_id'].isin(user_id_g20)]
print(ieRS_ratings_g20.shape)
    # (15056975, 4)
ieRS_ratings_g20.to_csv('./data/ieRS_ratings_g20.csv', index = False)

movid_id_g20 = ieRS_ratings_g20.movie_id.unique()
print(len(movid_id_g20))
# 9064

ieRS_movieInfo_emotions_g20 = intersection[intersection['movie_id'].isin(movid_id_g20)]
    # 9064 movies
    # ['movie_id', 'imdb_id', 'title(year)', 'title', 'year', 'runtime', 'genre', 'aveRating', 'director', 'writer', 'description', 'cast', 'poster', ]
ieRS_movieInfo_emotions_g20.to_csv('./data/ieRS_movieInfo_emotions_g20.csv', index= False)
# print(ieRS_movieInfo_emotions_g20.head())

ieRS_emotions_g20 = emotions[emotions['movie_id'].isin(movid_id_g20)] 
# 9064 movies
ieRS_emotions_g20.to_csv('./data/ieRS_emotions_g20.csv', index= False)










