"""
compute.py
"""
# 6/4/2022 updated by Lijie Guo
# There are 5 lines of notes for Shanhan starting with "##!!! For Shahan:"
from typing import List
from models import Rating, Recommendation, Preference, LatentFeature, EmotionalSignature, EmoButtonInput
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

import ieRSalgs.ieRS_recommender as ieRS
import ieRSalgs.diversification as div
import ieRSalgs.setpath as setpath

import pandas as pd
import numpy as np
import os

def get_RSSA_preds(ratings: List[Rating], user_id) -> pd.DataFrame:
    rated_items = np.array([np.int64(rating.movielensId) for rating in ratings])
    new_ratings = pd.Series(np.array([np.float64(rating.rating) for rating in ratings]), index = rated_items)    

    data_path = os.path.join(os.path.dirname(__file__), './ieRSalgs/data/ieRS_item_popularity.csv')
    eRS_item_popularity = pd.read_csv(data_path) 
    
    model_path = os.path.join(os.path.dirname(__file__), './ieRSalgs/data/ieRS_implictMF.pkl')
    trained_MF_model = ieRS.import_trained_model(model_path)

    ## predicting
    [RSSA_preds, liveUser_feature] = ieRS.live_prediction(trained_MF_model, user_id, new_ratings, eRS_item_popularity)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # liveUser_feature: np.ndarray
    # extract the not-rated-yet items
    RSSA_preds_of_noRatedItems = RSSA_preds[~RSSA_preds['item'].isin(rated_items)]
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    
    return RSSA_preds_of_noRatedItems, trained_MF_model
        # return trained_MF_model since it will be needed in the diversification to extract the latent features

def sqrt_cityblock(point1, point2):
    sqrt_city_block = 0
    for i in range(len(point1)):
        dist_axis = abs(point2[i] - point1[i])
        sqrt_city_block +=  dist_axis**(1/2)

    return sqrt_city_block 

def emotion_distance(matrix, vector):
    dist_array = []
    matrix_max = np.max(matrix, axis = 0)# added
    matrix_min = np.min(matrix, axis = 0)# added
    scaled_vector = (matrix_max - matrix_min)*vector# added
    for row_vector in matrix:
        # dist = distance.cityblock(row_vector, vector)
        # dist = sqrt_cityblock(row_vector, vector)
        dist = distance.euclidean(row_vector, scaled_vector)
            # change vector - scaled_vector
        dist_array.append(dist)
    
    return  dist_array 
    
def weighted_ranking(original_rec_df, specified_emotion_vals, specified_emotion_tags, item_emotions_df):
    # original_rec_df: top-N items, ['item', 'discounted_score'] in ieRS project
    # specified_emotion_vals: 1-d array, each element corresponds to specified_emotion_tags
    # specified_emotion_tags: emotion labels, each label corresponds to specified_emotion_vals
    # item_emotions_df: the emotional signature of the movies in original_rec['item']
        # ['item', 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    
    ## create 'ori_rank' to mark the initial ranking
    original_rec_df.insert(original_rec_df.shape[1], 'ori_rank', range(0, original_rec_df.shape[0]))
        # ['item', 'discounted_score', 'ori_rank']
    
    ## merge the initial ranking and emotional signatures
    recs_emotions_df = pd.merge(original_rec_df, item_emotions_df, on = 'item')
    candidates_df_toScale = recs_emotions_df[['ori_rank', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']]
    
    ## do Max-Min scaling on the initial ranking and the 8 emotional signatures, item IDs keep the same
    scaler = MinMaxScaler()
    candidates_df_scared = scaler.fit_transform(candidates_df_toScale.to_numpy())
    candidates_df_scared = pd.DataFrame(candidates_df_scared, columns=['ori_rank', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'])    
    # candidates_df_scared.insert(0, 'item', recs_emotions_df['item'].values)
        # ['item', 'ori_rank', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    
    # calculate the new ranking
    new_ranking_score = np.sum(candidates_df_scared[specified_emotion_tags].values * specified_emotion_vals, axis = 1) + (1-np.sum(np.absolute(specified_emotion_vals)))*candidates_df_scared['ori_rank'].values
    recs_emotions_df.insert(recs_emotions_df.shape[1], 'new_rank_score', new_ranking_score)
        # ['item', 'discounted_score', 'ori_rank', ...
        # ..., 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', ...
        # ..., 'new_rank_score']
    
    return recs_emotions_df



    
#1a. Traditional top-N  
def predict_topN(ratings: List[Rating], user_id) -> List[str]:
    [RSSA_preds_of_noRatedItems, _] = get_RSSA_preds(ratings, user_id)
    # traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        # only needed when using the traditional predictions
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # ** only contains the unrated items
        
    numRec = 10         
    recs_topN_discounted = discounted_preds_sorted.head(numRec)    
    
    recommendations = []
    for index, row in recs_topN_discounted.iterrows():
        recommendations.append(Recommendation(str(np.int64(row['item']))))
        
    return recommendations

    
    
#1b. Traditional top-N + Taking inputs
def predict_tuned_topN(ratings: List[Rating], user_id, emotion_input: List[EmoButtonInput]) -> List[str]:
    [RSSA_preds_of_noRatedItems, _] = get_RSSA_preds(ratings, user_id)
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # ** only contains the unrated items 
    num_topN = 200        
    topN_discounted = discounted_preds_sorted.head(num_topN)
    ##!!! For Shahan: the above 6 line of code(including the commented lines) are repeated for testing 
    
    #Assume: get the whole 8-dimension vector of the emotion, order:
        #anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    emotion_tags = [one_emotion.emotion for one_emotion in emotion_input]
    # print(type(emotion_tags))
    # print(emotion_tags)
    user_emotion_vals = [one_emotion.weight for one_emotion in emotion_input]
        # 8 * 1
    # print(emotion_tags)
    
    user_emotion_dict = dict(zip(emotion_tags, user_emotion_vals))
    #print(user_emotion_dict)
    
    user_specified_emotion_tags = []
    user_unspecified_emotion_tags = []
    user_specified_emotion_vals = []
    # user_unspecified_emotion_vals = []
    for k, v in user_emotion_dict.items():
        if v == "low":
            user_specified_emotion_tags.append(k)
            user_specified_emotion_vals.append(-0.125)
        elif v == "high":
            user_specified_emotion_tags.append(k)
            user_specified_emotion_vals.append(0.125)
        else: 
            user_unspecified_emotion_tags.append(k)
            # user_unspecified_emotion_vals.append(k)     
    # print(user_specified_emotion_tags)
    # print(user_specified_emotion_vals)  
    # print(user_unspecified_emotion_tags)      
        
    ##!!! For Shahan: start to process the top-200 movies by extracting the "unspecified" emotions for distance calculation and re-ording
    ##Show movies (among the top-200) with high values on the specified emotions
    item_emotions_filename = os.path.join(os.path.dirname(__file__), './ieRSalgs/data/ieRS_emotions_g20.csv')
    item_emotions = pd.read_csv(item_emotions_filename) 
        # ['movie_id', 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    item_emotions = item_emotions.rename({'movie_id' : 'item'}, axis = 1)
    
    candidate_ids = topN_discounted.item.unique()
    candidate_item_emotions_df = item_emotions[item_emotions['item'].isin(candidate_ids)]
        # ['item', 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    # print(candidate_item_emotions_df.shape)
        # 200, 10
    
    ## Call the weighted_ranking() function
    new_ranking_score_df = weighted_ranking(topN_discounted[['item', 'discounted_score']], user_specified_emotion_vals, user_specified_emotion_tags, candidate_item_emotions_df)
    new_ranking_score_df_sorted = new_ranking_score_df.sort_values(by = 'new_rank_score', ascending = False)
    
    numRec = 10
    recs_topN_reRanked = new_ranking_score_df_sorted.head(numRec)
    
    recommendations = []
    for index, row in recs_topN_reRanked.iterrows():
        recommendations.append(Recommendation(str(np.int64(row['item']))))
        
    return recommendations
        
    
    
    
    
#2a. diversification by emotion
def predict_diverseN_by_emotion(ratings: List[Rating], user_id) -> List[str]:
    [RSSA_preds_of_noRatedItems, _] = get_RSSA_preds(ratings, user_id)
    # traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        # only needed when using the traditional predictions
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score'] 
        # ** only contains the unrated items
    
    ## diversified by emotion
    num_topN = 200
    candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    ## recommendation
    item_emotions_filename = os.path.join(os.path.dirname(__file__), './ieRSalgs/data/ieRS_emotions_g20.csv')
    item_emotions = pd.read_csv(item_emotions_filename) 
        # ['movie_id', 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    item_emotions = item_emotions.rename({'movie_id' : 'item'}, axis = 1)
    item_emotions_ndarray = item_emotions[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']].to_numpy()
        # np.ndarray of 8 columns matches ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    item_ids = item_emotions.item.unique()
        # double_checked: item_ids is not sorted
        # but if the unique() is called with return_counts = True, then the returns will be sorted by the unique elements by default
    # print(item_emotions.head(20))
    # print(item_ids[:20])
    
    numRec = 10         
    weighting = 0
    [rec_diverseEmotion, rec_itemEmotion] = div.diversify_item_feature(candidates, item_emotions_ndarray, item_ids, weighting, numRec)
        # rec_diverseEmotion: ['item', 'discounted_score']
        # rec_itemEmotion : ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    
    recommendations = []
    for index, row in rec_diverseEmotion.iterrows():
        recommendations.append(Recommendation(str(np.int64(row['item']))))
 
    return recommendations
    
  
#2b. Diversification by emotion + Taking inputs
def predict_tuned_diverseN_by_emotion(ratings: List[Rating], user_id, emotion_input: List[EmoButtonInput]) -> List[str]:
    [RSSA_preds_of_noRatedItems, _] = get_RSSA_preds(ratings, user_id)
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # ** only contains the unrated items 

    ## diversified by emotion
    num_topN = 200
    candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    ## recommendation
    item_emotions_filename = os.path.join(os.path.dirname(__file__), './ieRSalgs/data/ieRS_emotions_g20.csv')
    item_emotions = pd.read_csv(item_emotions_filename) 
        # ['movie_id', 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    item_emotions = item_emotions.rename({'movie_id' : 'item'}, axis = 1)
    
    item_emotions_ndarray = item_emotions[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']].to_numpy()
        # np.ndarray of 8 columns matches ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    item_ids = item_emotions.item.unique()
        # double_checked: item_ids is not sorted
        # but if the unique() is called with return_counts = True, then the returns will be sorted by the unique elements by default
    # print(item_emotions.head(20))
    # print(item_ids[:20])   
    numDiv = 100         
    weighting = 0
    [TopDiverseEmotion, rec_itemEmotion] = div.diversify_item_feature(candidates, item_emotions_ndarray, item_ids, weighting, numDiv)
        # TopDiverseEmotion: ['item', 'discounted_score']
        # rec_itemEmotion : ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

    #Assume: get the whole 8-dimension vector of the emotion, order:
        #anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    emotion_tags = [one_emotion.emotion for one_emotion in emotion_input]
    # print(type(emotion_tags))
    # print(emotion_tags)
    user_emotion_vals = [one_emotion.weight for one_emotion in emotion_input]
        # 8 * 1
    # print(emotion_tags)
    user_emotion_dict = dict(zip(emotion_tags, user_emotion_vals))
    #print(user_emotion_dict)
    
    user_specified_emotion_tags = []
    user_unspecified_emotion_tags = []
    user_specified_emotion_vals = []
    # user_unspecified_emotion_vals = []
    for k, v in user_emotion_dict.items():
        if v == "low":
            user_specified_emotion_tags.append(k)
            user_specified_emotion_vals.append(-0.125)
        elif v == "high":
            user_specified_emotion_tags.append(k)
            user_specified_emotion_vals.append(0.125)
        else: 
            user_unspecified_emotion_tags.append(k)
            # user_unspecified_emotion_vals.append(k)     
    # print(user_specified_emotion_tags)
    # print(user_specified_emotion_vals)
    # print(user_unspecified_emotion_tags)  
      
    candidate_diversedRec_ids = TopDiverseEmotion.item.unique()
    candidate_diversedRec_item_emotions_df = item_emotions[item_emotions['item'].isin(candidate_diversedRec_ids)]
        # ['item', 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    # print(candidate_diversedRec_item_emotions_df.shape)
        # 50, 10, depends on numDiv
    
    ## Call the weighted_ranking() function
    new_ranking_diverseEmotion_df = weighted_ranking(TopDiverseEmotion[['item', 'discounted_score']], user_specified_emotion_vals, user_specified_emotion_tags, candidate_diversedRec_item_emotions_df)
    new_ranking_diverseEmotion_df_sorted = new_ranking_diverseEmotion_df.sort_values(by = 'new_rank_score', ascending = False)

    numRec = 10
    rec_new_ranking_diverseEmotion = new_ranking_diverseEmotion_df_sorted.head(numRec)
    
    recommendations = []
    for index, row in rec_new_ranking_diverseEmotion.iterrows():
        recommendations.append(Recommendation(str(np.int64(row['item']))))
        
    return recommendations



if __name__ == '__main__':

    ### Pull ratings
    liveUserID = 'Bart'
    fullpath_test = os.path.join(os.path.dirname(__file__), './ieRSalgs/testing_rating_rated_items_extracted/ratings_set6_rated_only_' + liveUserID + '.csv')
    ratings_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
    #print(ratings_liveUser.head(20))
    
    ratings = []
    for index, row in ratings_liveUser.iterrows():
        ratings.append(Rating(row['item'], row['rating']))
    
    ### Pull emotion inputs
    fullpath_test = os.path.join(os.path.dirname(__file__), './ieRSalgs/testing_rating_rated_items_extracted/emotion_button_input_' + 'Shahan' + '.csv')
    emotion_input_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
    # print(emotion_input_liveUser)
      
    input = []
    for index, row in emotion_input_liveUser.iterrows():
        input.append(EmoButtonInput(row['emotion'], row['weight']))
    # print(input)
    
    ### start recommending
    recommendations = predict_topN(ratings, liveUserID)
    # print('1a - Traditional top-N recommendations')
    # print(recommendations)
    for rec in recommendations:
        print(rec.movielensId, end = ', ')
    print()  
      
    
    
    recommendations = predict_tuned_topN(ratings, liveUserID, input)
    # print('1b - Recommendations after taking the emotion input from the topN condition')
    # print(recommendations)
    for rec in recommendations:
        print(rec.movielensId, end = ', ')
    print()
    
    
    recommendations = predict_diverseN_by_emotion(ratings, liveUserID)
    # print('2a - Diversified by emotions')
    # print(recommendations)
    for rec in recommendations:
        print(rec.movielensId, end = ', ')   
    print()
    
    
    recommendations = predict_tuned_diverseN_by_emotion(ratings, liveUserID, input)
    # print('2b - Recommendations after taking the emotion input from the diverseN condition')
    # print(recommendations)
    for rec in recommendations:
        print(rec.movielensId, end = ', ')
    print()  

    

        
  
    
    

    '''
    RSSA_team = ['Bart', 'Sushmita', 'Shahan', 'Aru', 'Mitali', 'Yash']
    for liveUserID in RSSA_team:
        fullpath_test = os.path.join(os.path.dirname(__file__), 'ieRSalgs/testing_rating_rated_items_extracted/ratings_set6_rated_only_' + liveUserID + '.csv')
        ratings_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
        ratings = []
        for index, row in ratings_liveUser.iterrows():
            ratings.append(Rating(row['item'], row['rating']))
        recommendations = predict_user_topN(ratings, liveUserID)
        for rec in recommendations:
            print(rec.movielensId, end = ', ')
        print()
    '''
    
    