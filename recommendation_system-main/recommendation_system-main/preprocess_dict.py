import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

print("Reading the dataset...")
dataset = pd.read_csv('./data/small_rating.csv')

N = dataset.userId.max() + 1
M = dataset.movie_idx.max() + 1
print("Total users: ", N, " Total Movies: ", M)



print("Separating the dataset into training and test set...")
dataset = shuffle(dataset)
cutoff = int(0.8 * len(dataset))
dataset_train = dataset.iloc[:cutoff]
dataset_test = dataset.iloc[cutoff:]



user2movie = {}
movie2user = {}
usermovie2rating =  {}

print("Calling: update_user2movie_and_movie2user...")
count = 0
def update_user2movie_and_movie2user(row) :
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/cutoff))
    i = int(row.userId)
    j = int(row.movie_idx)
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)

    usermovie2rating[(i, j)] = row.rating
dataset_train.apply(update_user2movie_and_movie2user, axis=1)


usermovie2rating_test = {}
print("Calling: update_usermovie2rating_test...")
count = 0
def update_usermovie2rating_test(row):
  global count
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/len(dataset_test)))

  i = int(row.userId)
  j = int(row.movie_idx)
  usermovie2rating_test[(i,j)] = row.rating
dataset_test.apply(update_usermovie2rating_test, axis=1)


# Not actually json! A binary format

with open('user2movie.json', 'wb') as f:
  pickle.dump(user2movie, f)

with open('movie2user.json', 'wb') as f:
  pickle.dump(movie2user, f)

with open('usermovie2rating.json', 'wb') as f:
  pickle.dump(usermovie2rating, f)

with open('usermovie2rating_test.json', 'wb') as f:
  pickle.dump(usermovie2rating_test, f)
