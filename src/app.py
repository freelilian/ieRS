"""

app.py

Lijie Guo
Clemson University
04/06/2022

Server for running the recommender algorithms. See
`models.py` for information about the input and
outputs.

"""

from pathlib import Path
import json

from flask import Flask, abort
from flask import request
from flask import render_template

from compute_slider import predict_topN, predict_tuned_topN, predict_diverseN_by_emotion, predict_tuned_diverseN_by_emotion
from models import Rating, EmoSliderInput

# from compute_button import predict_topN, predict_tuned_topN, predict_diverseN_by_emotion, predict_tuned_diverseN_by_emotion
# from models import Rating, EmoButtonInput

app = Flask(__name__)


@app.route('/')
def show_readme():
    return render_template('README.html')

@app.route('/initial_recs', methods=['POST'])
def predict_preferences():
    req = request.json
    ratings = None

    try:
        ratings = req['user_ratings']
    except KeyError:
        abort(400)
        
    ratings = [Rating(**rating_entry) for rating_entry in ratings]    

    rec_funcs = {
        'topN': predict_topN,
        'diverseN': predict_diverseN_by_emotion,
    }
    recommendations = {k: f(ratings=ratings, user_id=0) for k, f in rec_funcs.items()}
        
    return dict(preferences=recommendations)

@app.route('/tuned_recs', methods=['POST'])
def predict_tuned_preferences():
    req = request.json
    ratings = None

    try:
        ratings = req['user_ratings']
        emotion_input = req['emotion_input']
    except KeyError:
        abort(400)
        
    ratings = [Rating(**rating_entry) for rating_entry in ratings]  
    emo_input = [EmoSliderInput(**emo_entry) for emo_entry in emotion_input]
    # emo_input = [EmoButtonInput(**emo_entry) for emo_entry in emotion_input] 

    tuned_rec_funcs = {
        'tuned_topN': predict_tuned_topN,
        'tuned_diverseN': predict_tuned_diverseN_by_emotion,
    }
    
    recommendations = {k: f(ratings=ratings, user_id=0, emotion_input = emo_input) for k, f in tuned_rec_funcs.items()}
    
    return dict(preferences=recommendations)
    




if __name__ == '__main__':
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path) as f:
        settings = json.load(f)
    app.run(port=settings['port'],
            host=settings['host'],
            debug=settings['debug'])
