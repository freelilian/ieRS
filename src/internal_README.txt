Since .pkl also save the path of the trained MF model, so it needs to re-train the 
model with different relative paths specified in the dependencies in the train_mf models;

Also, compute.py call the functions under the folder eRSalgs, we also need to be careful 
about the pathes in the dependencies in the compute.py and the eRS_recommender.py (and 
diversification.py, but it does not call any self-definded functions in this case.)

When running the ieRS_recommendations.py and main.py, do not need to change the path, just
be careful with the trained MF model which should be a model trained without 'from . import blabla...'