"""
compute.py
"""
# 6/4/2022 updated by Lijie Guo
# There are 5 lines of notes for Shanhan starting with "##!!! For Shahan:"
from typing import List
from models import Rating, Recommendation, Preference, LatentFeature, EmotionalSignature, EmoSliderInput
from scipy.spatial import distance

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
    for row_vector in matrix:
        # dist = distance.cityblock(row_vector, vector)
        # dist = sqrt_cityblock(row_vector, vector)
        dist = distance.euclidean(row_vector, vector)
        dist_array.append(dist)
    
    return  dist_array 
    
       
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
def predict_tuned_topN(ratings: List[Rating], user_id, emotion_input: List[EmoSliderInput]) -> List[str]:
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
    user_emotion_switch = [one_emotion.switch for one_emotion in emotion_input]
    user_emotion_vals = np.array([np.float(one_emotion.weight) for one_emotion in emotion_input])
        # 8 * 1
    # print(emotion_tags)
    
    user_emotion_switch_dict = dict(zip(emotion_tags, user_emotion_switch))
    user_emotion_vals_dict = dict(zip(emotion_tags, user_emotion_vals))
    #print(user_emotion_vals_dict)
    
    user_specified_emotion_tags = []
    user_unspecified_emotion_tags = []
    user_specified_emotion_vals = []
    # user_unspecified_emotion_vals = []
    for k, v in user_emotion_switch_dict.items():
        if v == "specified":
            user_specified_emotion_tags.append(k)
            user_specified_emotion_vals.append(user_emotion_vals_dict[k])
        else: 
            user_unspecified_emotion_tags.append(k)
            # user_unspecified_emotion_vals.append(k)     
    # print(user_specified_emotion_tags)
    # print(user_unspecified_emotion_tags)      

    ##!!! For Shahan: start to process the top-200 movies by extracting the "unspecified" emotions for distance calculation and re-ording
    ##Show movies (among the top-200) with high values on the specified emotions
    item_emotions_filename = os.path.join(os.path.dirname(__file__), './ieRSalgs/data/ieRS_emotions_g20.csv')
    item_emotions = pd.read_csv(item_emotions_filename) 
        # ['movie_id', 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    item_emotions = item_emotions.rename({'movie_id' : 'item'}, axis = 1)
    
    candidate_ids = topN_discounted.item.unique()
    candidate_item_emotions = item_emotions[item_emotions['item'].isin(candidate_ids)]
    # print(candidate_item_emotions.shape)
        # 200, 10
    # print('checked shoune 200 items')
    # print(candidate_item_emotions)
    candidate_item_specified_emotions_ndarray = candidate_item_emotions[user_specified_emotion_tags].to_numpy()
        # np.ndarray of specified columns matches user_specified_emotion_tags
        # for distance calculation
    # print(candidate_item_emotions[user_specified_emotion_tags].head(10))
    candidate_item_ids = candidate_item_emotions.item.unique()
    
    distance_to_input = emotion_distance(candidate_item_specified_emotions_ndarray, user_specified_emotion_vals)
    distance_to_input_df = pd.DataFrame({'item': candidate_item_ids, 'distance': distance_to_input}, columns = ['item', 'distance'])
    distance_to_input_df_sorted = distance_to_input_df.sort_values(by = 'distance', ascending = False)
    numRec = 10
    rec_distance_to_input_df_sorted = distance_to_input_df_sorted.head(numRec)
    
    recommendations = []
    for index, row in rec_distance_to_input_df_sorted.iterrows():
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
def predict_tuned_diverseN_by_emotion(ratings: List[Rating], user_id, emotion_input: List[EmoSliderInput]) -> List[str]:
    [RSSA_preds_of_noRatedItems, _] = get_RSSA_preds(ratings, user_id)
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # ** only contains the unrated items 
    num_topN = 200        
    topN_discounted = discounted_preds_sorted.head(num_topN)
    ##!!! For Shahan: the above 6 line of code(including the commented lines) are repleated for testing
    
    #Assume: get the whole 8-dimension vector of the emotion, order:
        #anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    emotion_tags = [one_emotion.emotion for one_emotion in emotion_input]
    # print(type(emotion_tags))
    # print(emotion_tags)
    user_emotion_switch = [one_emotion.switch for one_emotion in emotion_input]
    user_emotion_vals = np.array([np.float(one_emotion.weight) for one_emotion in emotion_input])
        # 8 * 1
    # print(emotion_tags)
    
    
    user_emotion_switch_dict = dict(zip(emotion_tags, user_emotion_switch))
    user_emotion_vals_dict = dict(zip(emotion_tags, user_emotion_vals))
    #print(user_emotion_vals_dict)
    
    user_specified_emotion_tags = []
    user_unspecified_emotion_tags = []
    user_specified_emotion_vals = []
    # user_unspecified_emotion_vals = []
    for k, v in user_emotion_switch_dict.items():
        if v == "specified":
            user_specified_emotion_tags.append(k)
            user_specified_emotion_vals.append(user_emotion_vals_dict[k])
        else: 
            user_unspecified_emotion_tags.append(k)
            # user_unspecified_emotion_vals.append(k)     
    # print(user_specified_emotion_tags)
    # print(user_unspecified_emotion_tags)  
       
    ##!!! For Shahan: start to process the top-200 movies by extracting the "unspecified" emotions for diversification and re-ording
    ##Show movies (among the top-200) with high values on the specified emotions
    item_emotions_filename = os.path.join(os.path.dirname(__file__), './ieRSalgs/data/ieRS_emotions_g20.csv')
    item_emotions = pd.read_csv(item_emotions_filename) 
        # ['movie_id', 'imdb_id', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    item_emotions = item_emotions.rename({'movie_id' : 'item'}, axis = 1)
    
    candidate_ids = topN_discounted.item.unique()
    candidate_item_emotions = item_emotions[item_emotions['item'].isin(candidate_ids)]
    # print(candidate_item_emotions.shape)
        # 200, 10
    # print('checked shoune 200 items')
    # print(candidate_item_emotions)
    candidate_item_specified_emotions_ndarray = candidate_item_emotions[user_specified_emotion_tags].to_numpy()
        # np.ndarray of specified columns matches user_specified_emotion_tags
        # for distance calculation
    candidate_item_unspecified_emotions_ndarray = candidate_item_emotions[user_unspecified_emotion_tags].to_numpy()
        # np.ndarray of specified columns matches user_unspecified_emotion_tags
        # for diversification
    # print(candidate_item_emotions[user_specified_emotion_tags].head(10))
    candidate_item_ids = candidate_item_emotions.item.unique()
    
    distance_to_input = emotion_distance(candidate_item_specified_emotions_ndarray, user_specified_emotion_vals)
    distance_to_input_df = pd.DataFrame({'item': candidate_item_ids, 'distance': distance_to_input}, columns = ['item', 'distance'])
    distance_to_input_df_sorted = distance_to_input_df.sort_values(by = 'distance', ascending = False)
    # candidates for diversification
    candidates_for_div = distance_to_input_df_sorted
    # ['item', 'distance']

    numRec = 10         
    weighting = 0
    ##!!! For Shahan: The diversification algorithm got called here to reorder the top-200 movies
    [rec_diverseEmotion, rec_itemEmotion] = div.diversify_item_feature(candidates_for_div, candidate_item_unspecified_emotions_ndarray, candidate_item_ids, weighting, numRec)
        # rec_diverseEmotion: ['item', 'distance']
        # rec_itemEmotion : user_unspecified_emotion_tags
    
    recommendations = []
    for index, row in rec_diverseEmotion.iterrows():
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
    fullpath_test = os.path.join(os.path.dirname(__file__), './ieRSalgs/testing_rating_rated_items_extracted/emotion_slider_input_' + 'Shahan' + '.csv')
    emotion_input_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
    # print(emotion_input_liveUser)
      
    input = []
    for index, row in emotion_input_liveUser.iterrows():
        input.append(EmoSliderInput(row['emotion'], row['switch'], row['weight']))
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
    
    