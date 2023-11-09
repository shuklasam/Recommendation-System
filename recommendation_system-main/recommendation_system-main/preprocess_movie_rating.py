import pandas as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.metrics.pairwise import cosine_similarity


dataset_ratings = pd.read_csv('./large-datasets/archive/edited_rating.csv')
dataset_movies = pd.read_csv('./data/movies.csv')

movie_avg_ratings = dataset_ratings.groupby(
    'movieId')['rating'].mean().to_dict()

dataset_movies['movie_rating'] = dataset_movies.apply(
    lambda row: movie_avg_ratings.get(row['movie_idx'], 0.5), axis=1)

genre_strings = dataset_movies['movie_genre'].str.findall(r"'([^']+)'")
unique_genres = set()

# Iterate through each list of genres and add genres to the set
for genres_list in genre_strings:
    unique_genres.update(genres_list)

# print(unique_genres)

genreId = {}
count = 0
for genre in unique_genres:
    genreId[genre] = count
    count += 1

dataset_movies['movie_genre'] = dataset_movies['movie_genre'].apply(
    lambda x: eval(x))
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(
    dataset_movies['movie_genre']), columns=mlb.classes_, index=dataset_movies.index)
scaler = StandardScaler()
scaled_genre_encoded = pd.DataFrame(scaler.fit_transform(
    genre_encoded), columns=genre_encoded.columns, index=genre_encoded.index)

# Combine the scaled genre columns back with the original DataFrame
data = pd.concat([dataset_movies, scaled_genre_encoded], axis=1)


data = data.drop(
    columns=['movie_title', 'movie_genre', '(no genres listed)'], axis=1)


# cosine_sim_matrix = cosine_similarity(data)
# print(cosine_sim_matrix)


rated_movied_dataset = pd.read_csv('./data/movies_rated.csv')

def getSimilarMatrix(movieId):
    target_movie_idx = movieId
    cosine_sim_scores = cosine_similarity([data.iloc[target_movie_idx]], data)
    similarity_scores = list(enumerate(cosine_sim_scores[0]))
    sorted_similar_movies = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)
    top_n = 10
    sim_movies = []
    for i, score in sorted_similar_movies[1:top_n+1]:
        selected_row = rated_movied_dataset[rated_movied_dataset['movie_idx'] == i]
        recom_movie = {}
        recom_movie['genre'] = selected_row.movie_genre.values[0]
        recom_movie['title'] = selected_row.movie_title.values[0]
        recom_movie['year'] = selected_row.movie_year.values[0]
        recom_movie['rating'] = selected_row.movie_rating.values[0]
        recom_movie['id'] = selected_row.movie_idx.values[0]
        sim_movies.append(recom_movie)
    

    return sim_movies


# print(getSimilarMatrix(1))


# data.to_csv('./data/movies_rated_encoded.csv', index=False)
