"""
test.py
"""
import requests
import json

host = 'http://127.0.0.1:5000'
with open('test_ratings.json') as f:
    data1 = json.load(f)
#host = 'http://130.127.106.99:5003'

endpoint = 'initial_recs'
res = requests.post(f' {host}/{endpoint}', json=dict(user_ratings=data1['ratings'], user_id=None))
print("Round 1")
print(res.text)

print("Round 2")
with open('test_emotion_slider_input.json') as f:
# with open('test_emotion_button_input.json') as f:
    data2 = json.load(f)
    
endpoint = 'tuned_recs'
res = requests.post(f' {host}/{endpoint}', json=dict(user_ratings=data1['ratings'], user_id=None, emotion_input=data2['input']))
print(res.text)