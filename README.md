# ieRS Algorithms

## Overview

This directory contains code for running the recommender algorithms server. This server has been
separated from the main server to isolate large dependencies. 
The `/initial_recs` endpoint accepts a series of ratings and outputs either top-N/diverse-N items. 
The `/tuned_recs` endpoint accepts a series of ratings & emotion inputs, and outputs either 
tuned-top-N/tuned-diverse-N items.
See `tests/test_ratings.json`, `test_emotion_button_input.json` and `test_emotion_slider_input.json`  
for example ratings and emotion input schemas.

## Usage

Start by installing all the dependencies (it is recommended to use `conda`):


|    Type     |        Location             |
|-------------|-----------------------------|
| Algorithms  |  src/ieRSalgs/lenskit11.yml |
| Server      |  requirements.txt           |
| Testing     |  tests/                     |

Then configure `src/config.json` 
and finally start the server with `python src/app.py`. 

## Data files
Please download the pre-trained data from 
https://drive.google.com/drive/folders/1Fv7RVFrBA-76ZLlDItlybsHxS5iS8gh_?usp=sharing,
and place them in this location: src/ieRSalgs/data/

Please download the testing data from 
https://drive.google.com/drive/folders/1_6ZURIj0x8rGAwBLsgNFIjfcu5QibkdH?usp=sharing,
and place them in this location: src/ieRSalgs/testing_rating_rated_items_extracted/