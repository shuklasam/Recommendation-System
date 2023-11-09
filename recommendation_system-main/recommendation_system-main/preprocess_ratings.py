import pandas as pd


print("reading dataset...")
dataset = pd.read_csv('./large-datasets/archive/rating.csv')
print("Got all the ratings...")


print("Starting Data preprocessing...")
print("changing user ids range...")
dataset.userId = dataset.userId - 1
print("user ids changed to the range of 0...N-1")

print("rewritting the movies id...")
unique_movie_ids = set(dataset.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1

dataset['movie_idx'] = dataset['movieId'].map(movie2idx)
print("movies ids changed to the range of 0 to len(unique_movies)")


print("removing timestamps...")
dataset = dataset.drop(columns=['timestamp'])

print("rewriting the data to a new file...")
dataset.to_csv('./large-datasets/archive/edited_rating.csv', index=False)

print("Preprocessing Successful! Data preprocessed...")