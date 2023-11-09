import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

import os
if not os.path.exists('user2movie.json') or \
   not os.path.exists('movie2user.json') or \
   not os.path.exists('usermovie2rating.json') or \
   not os.path.exists('usermovie2rating_test.json'):
    import preprocess_dict
with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)



movies_database = pd.read_csv('./data/movies.csv')


N = np.max(list(user2movie.keys())) + 1
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N: ", N, "M: ", M)

K = 20  # number of neighbors we'd like to consider
limit = 2  # number of common movies users must have in common in order to consider
neighbors = []  # store neighbors in this list
averages = []  # each item's average rating
deviations = []  # each item's deviation

print("parsing products...")
for i in range(M):
    # find the K closest items to item i
    users_i = movie2user[i]
    users_i_set = set(users_i)

    # calculate avg and deviation
    ratings_i = {user: usermovie2rating[(user, i)] for user in users_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {user: (rating - avg_i) for user, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    # save these for later use
    averages.append(avg_i)
    deviations.append(dev_i)

    sl = SortedList()
    for j in range(M):
        # don't include yourself
        if j != i:
            users_j = movie2user[j]
            users_j_set = set(users_j)
            common_users = (users_i_set & users_j_set)  # intersection
            if len(common_users) > limit:
                # calculate avg and deviation
                ratings_j = {user: usermovie2rating[(
                    user, j)] for user in users_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = {user: (rating - avg_j)
                         for user, rating in ratings_j.items()}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                # calculate correlation coefficient
                numerator = sum(dev_i[m]*dev_j[m] for m in common_users)
                w_ij = numerator / (sigma_i * sigma_j)

                # insert into sorted list and truncate
                # negate weight, because list is sorted ascending
                # maximum value (1) is "closest"
                sl.add((-w_ij, j))
                if len(sl) > K:
                    del sl[-1]

    # store the neighbors
    neighbors.append(sl)

# using neighbors, calculate train and test MSE

def predict(i, u):
    # calculate the weighted sum of deviations
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        # remember, the weight is stored as its negative
        # so the negative of the negative weight is the positive weight
        try:
            numerator += -neg_w * deviations[j][u]
            denominator += abs(neg_w)
        except KeyError:
            # neighbor may not have been rated by the same user
            # don't want to do dictionary lookup twice
            # so just throw exception
            pass

    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)  # min rating is 0.5
    return prediction


temp_user = 0
train_predictions = []
train_targets = []
for (u, m), target in usermovie2rating.items():
    # calculate the prediction for this movie
    prediction = predict(m, u)
    temp_user = u

    # save the prediction and target
    train_predictions.append(prediction)
    train_targets.append(target)

print("For debug/testing purpose you use the user with a uid of: " + str(temp_user))
print("Once site is live you can go to the {referred port in the terminal}/home/" + str(temp_user) + " to see recommendation for the user")


test_predictions = []
test_targets = []
# same thing for test set
for (u, m), target in usermovie2rating_test.items():
    # calculate the prediction for this movie
    prediction = predict(m, u)
    # print(prediction)

    # save the prediction and target
    test_predictions.append(prediction)
    test_targets.append(target)


# calculate accuracy
def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)


print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))

s2 = SortedList()
def top10Recommend(u):
    checked = set()
    for (user, movie), target in usermovie2rating.items():
        if (u == user):
            if (not movie in checked):
              prediction = predict(movie, u)
              checked.add(movie)
              s2.add((prediction, movie))
              if (len(s2) > 30):
                  del s2[0]
    recomendation = []
    for (rating, movieId) in s2:
        selected_row = movies_database[movies_database['movie_idx'] == movieId]
        recommended_movies = {}
        recommended_movies['genre'] = selected_row.movie_genre.values[0]
        recommended_movies['title'] = selected_row.movie_title.values[0]
        recommended_movies['year'] = selected_row.movie_year.values[0]
        recommended_movies['rating'] = rating
        recommended_movies['id'] = movieId
        recomendation.append(recommended_movies)
    return recomendation