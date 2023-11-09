import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re
import pickle


print("Reading dataset...")
dataset = pd.read_csv('large-datasets/archive/movie.csv')
print("Got all the ratings...")

print("Shuffling the dataset...")
dataset = shuffle(dataset)

tot_movies = 200
dataset_small = dataset.head(tot_movies)


movieOldId2NewId = {}
movieNewId2OldId = {}

count = 0 
for movieId in dataset_small['movieId']:
    movieOldId2NewId[movieId] = count
    movieNewId2OldId[count] = movieId
    count += 1

dataset_small.loc[:, 'movie_idx'] = dataset_small['movieId'].map(movieOldId2NewId).values

def extract_movie_title(input_string):
    pattern = r'^(.*?)\s*\((\d{4})\)$'
    match = re.match(pattern, input_string)
    
    if match:
        movie_title = match.group(1)
        # launch_year = int(match.group(2))
        return movie_title
    else:
        return None
def extract_movie_year(input_string):
    pattern = r'^(.*?)\s*\((\d{4})\)$'
    match = re.match(pattern, input_string)
    
    if match:
        # movie_title = match.group(1)
        launch_year = int(match.group(2))
        return launch_year
    else:
        return None


dataset_small.loc[:, 'movie_title'] = dataset_small['title'].map(lambda title: extract_movie_title(title)).values
dataset_small.loc[:, 'movie_year'] = dataset_small['title'].map(lambda title: extract_movie_year(title)).values
dataset_small.loc[:, 'movie_genre'] = dataset_small['genres'].map(lambda genres: genres.split('|')).values


with open('movieOldId2newId.json', 'wb') as f:
  pickle.dump(movieOldId2NewId, f)
with open('movieNewId2OldId.json', 'wb') as f:
  pickle.dump(movieNewId2OldId, f)

dataset_small = dataset_small.drop(columns=['title', 'movieId', 'genres'])


dataset_small.to_csv('./data/movies.csv', index=False)