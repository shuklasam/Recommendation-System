import pickle
import numpy as np
import pandas as pd

from collections import Counter

print("Reading the dataset...")
dataset = pd.read_csv('./large-datasets/archive/edited_rating.csv')
print("Original dataframe size: ", len(dataset))

N = dataset.userId.max() + 1
M = dataset.movie_idx.max() + 1


print("Counting user ids and movie ids...")
user_ids_count = Counter(dataset.userId)
movie_ids_count = Counter(dataset.movie_idx)


print("Shrinking the number of users to n = 10000 and number of movies m = 2000")
n = 1000
m = 200

user_ids = [u for u, c in user_ids_count.most_common(n)]
movie_ids = [m for m, c in movie_ids_count.most_common(m)]


print("Making a copy of the smaller dataset to avoid overwritting...")
dataset_small = dataset[dataset.userId.isin(user_ids) & dataset.movie_idx.isin(movie_ids)].copy()


print("Remaking (preprocessing) userIds as data no longer sequential...")
new_user_id_map = {}
i = 0
for old in user_ids:
    new_user_id_map[old] = i
    i += 1
print("i: ", i)
print("Remaking (preprocessing) movieIds as data no longer sequential...")
new_movie_id_map = {}
j = 0
for old in movie_ids:
  new_movie_id_map[old] = j
  j += 1
print("j: ", j)

print("Setting new IDs")
dataset_small.loc[:, 'userId'] = dataset_small['userId'].map(new_user_id_map)
dataset_small.loc[:, 'movie_idx'] = dataset_small['movie_idx'].map(new_movie_id_map)
print("max user id:", dataset_small.userId.max())
print("max movie id:", dataset_small.movie_idx.max())

print("Saving small dataframe...\nSmall dataframe size:", len(dataset_small))
dataset_small.to_csv('./data/small_rating.csv', index=False)

