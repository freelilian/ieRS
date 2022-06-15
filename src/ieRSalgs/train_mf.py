import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
# There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
        
import numpy as np
import pandas as pd
#from lenskit.algorithms import als
from . import customizedALS
from . import setpath
import time
import pickle
from .load_npz import load_trainset_npz
import os


    
data_path = setpath.setpath()
rating_file = data_path + 'ieRS_ratings_g20.csv'
ratings_train = pd.read_csv(rating_file)
    # ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_train = ratings_train.rename(columns = {'user_id': 'user', 'movie_id': 'item'})
    # rename the columns to ['user', 'item', 'rating', 'timestamp']
# print(ratings_train.head(10))

## 1 - Discounting the input ratings by ranking
# 1.1 - Calculating item rating counts and popularity rank,
# This will be used to discount the popular items from the input side
items, rating_counts = np.unique(ratings_train['item'], return_counts = True)
    # items is sorted by the unique elements by default
#>>> items = item_popularity.item.unique()
    # items is NOT sorted
items_rating_count = pd.DataFrame({'item': items, 'count': rating_counts}, columns = ['item', 'count'])
items_rating_count_sorted = items_rating_count.sort_values(by = 'count', ascending = False)
items_popularity = items_rating_count_sorted
items_popularity['rank'] = range(1, len(items_popularity)+1)
    # ['item', 'count', 'rank']

previsous_count = 0
previsous_rank = 0
for index, row in items_popularity.iterrows():
    current_count = row['count']
    
    if row['count'] == previsous_count:
        row['rank'] = previsous_rank

    previsous_count = current_count
    previsous_rank = row['rank']
            
#print(items_popularity.head(20))
#print(items_popularity.tail(20))
    # lowest rank = 476631
      
# 1.2 - Start to discounting the input ratings by ranking
b = 0.4
ratings_train_popularity = pd.merge(ratings_train, items_popularity, how = 'left', on = 'item')
ratings_train_popularity['discounted_rating'] = ratings_train_popularity['rating']*(1-b/(2*ratings_train_popularity['rank']))
ratings_train = ratings_train_popularity[['user', 'item', 'discounted_rating', 'timestamp']]
# print(ratings_train.head(10))
ratings_train = ratings_train.rename({'discounted_rating': 'rating'}, axis = 1)

## 2 - Train the implicit MF model
model_path = os.path.join(os.path.dirname(__file__), './data/')
f = open(model_path + 'ieRS_implictMF.pkl', 'wb')
print("Training MF models ...")
start = time.time()
# algo = als.ImplicitMF(20, iterations=10, method="lu")
algo = customizedALS.ImplicitMF(20, iterations=10, method="lu")

algo.fit(ratings_train)
end = time.time() - start
print("\nMF models trained.\n")
print('\nTime spent: %0.0fs' % end)
    # Time spent: 58s
print('\nExporting the trained model - an object - as a pkl file')
#f = open('./data/ieRS_implictMF.pkl', 'wb')
pickle.dump(algo, f)
f.close() 

print('\nSaving the item popularity as a csv file')
items_popularity.to_csv(data_path + 'ieRS_item_popularity.csv', index = False)
    # ['item', 'count', 'rank']

print('\nDone\n')