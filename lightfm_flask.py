#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)


plays = pd.read_csv('data/lastfm/user_artists.dat', sep='\t')
artists = pd.read_csv('data/lastfm/artists.dat', sep='\t', usecols=['id','name'])
ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
ap = ap.rename(columns={"weight": "playCount"})
artist_rank = ap.groupby(['name']) \
    .agg({'userID' : 'count', 'playCount' : 'sum'}) \
    .rename(columns={"userID" : 'totalUsers', "playCount" : "totalPlays"}) \
    .sort_values(['totalPlays'], ascending=False)
artist_rank['avgPlays'] = artist_rank['totalPlays'] / artist_rank['totalUsers']

ap = ap.join(artist_rank, on="name", how="inner").sort_values(['playCount'], ascending=False)
pc = ap.playCount
play_count_scaled = (pc - pc.min()) / (pc.max() - pc.min())
ap = ap.assign(playCountScaled=play_count_scaled)
ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')
ratings = ratings_df.fillna(0).values
sparsity = float(len(ratings.nonzero()[0])) / (ratings.shape[0] * ratings.shape[1]) * 100

X = csr_matrix(ratings)
n_users, n_items = ratings_df.shape
user_ids = ratings_df.index.values
artist_names = ap.sort_values("artistID")["name"].unique()

Xcoo = X.tocoo()
data = Dataset()
data.fit(np.arange(n_users), np.arange(n_items))
interactions, weights = data.build_interactions(zip(Xcoo.row, Xcoo.col, Xcoo.data)) 
train, test = random_train_test_split(interactions)

model = LightFM(learning_rate=0.05, loss='warp')
model.fit(train, epochs=10, num_threads=2)

# Generating the list of artists at start-up:
artIDs = ap['artistID'].unique()
numarts = len(ap['artistID'].unique())
listart = ""
for it, artName in enumerate(ap['name'].unique()):
    listart = listart + '<input type="checkbox" name="'+str(artIDs[it])+'" value="'+str(artName)+'">'+artName+'<br>'


# get_recommendation from Jupyter notebook:
def get_recommendation(userid, ratings=ratings):
    X = csr_matrix(ratings)
    svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=0)
    X_matrix_svd = svd.fit_transform(X)
    cosine_sim = linear_kernel(X_matrix_svd,X_matrix_svd[userid].reshape(1,-1))
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    user_indices = [i[0] for i in sim_scores]
    artist_top = []
    for i in user_indices:
        idx_i = np.argmax(ratings[i])
        if len(ap['name'][ap['artistID'] == idx_i].index)== 1:            
            if ap['name'][ap['artistID'] == idx_i].unique()[0] not in artist_top:
                artist_top.append( ap['name'][ap['artistID'] == idx_i].unique()[0] )
    return artist_top

# Default page (page 1):
@app.route('/')
def root():
    begin = '''
        <html><head></head><body>
            <div style="width:800px; margin:0 auto;">
            <h1><i>LightFM</i></h1>
            <br><form action="/search" method="post">
            <div style="width:1600px; margin:0 auto;justify-content: center;">
    '''
    end = '''
                <input type=submit value=search>
            </form><br></div></div>
        </body></html>
    '''
    return begin+listart+end

# Result page:
@app.route('/search', methods=['POST', 'GET'])
def search():
    begin = '''
        <html><head></head><body>
            <h1><a href="/">LightFM</a></h1>
    '''
    end = "</body></html>"
    if request.method == 'POST':
        res = ""
        newline = np.zeros(numarts) # Creating a new user
        for it in request.form:
            newline[int(it)] = 1 # mean() insteead of '1' would give more accurate results...
        ratings2 = np.vstack((ratings, newline)) # Adding the new user to the existing users
        recomm = get_recommendation(ratings2.shape[0]-1, ratings2)
        for art in recomm:
            res = res + "<p>"+str(art)+" "+"</p>"

        return begin + res + end
      
    else:
        return begin + end # in case user reloads result page.
