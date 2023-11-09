from flask import Flask, redirect, url_for, render_template, request
from item_based import top10Recommend
from preprocess_movie_rating import getSimilarMatrix
import pandas as pd
app = Flask(__name__)

movie_database = pd.read_csv('./data/movies.csv')
rating_database = pd.read_csv('./data/small_rating.csv')

@app.route('/')
def root():
    return render_template('index.html')


@app.route('/home/<int:uid>')
def home(uid):
    recommendation = top10Recommend(uid)
    print(recommendation)
    return render_template('home.html', data=recommendation)

@app.route('/movies/<int:id>')
def moviePage(id):
    selected_row = movie_database[movie_database['movie_idx'] == id]
    movie = {}
    movie['genre'] = selected_row.movie_genre.values[0]
    movie['title'] = selected_row.movie_title.values[0]
    movie['year'] = selected_row.movie_year.values[0]
    sim_movies = getSimilarMatrix(id)
    return render_template('movie.html', data=movie, similar=sim_movies)



@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if (request.method == 'GET'):
        return render_template('signup.html')
    if (request.method == 'POST'):
        redirect(url_for('home'))


@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if (request.method == 'GET'):
        return render_template('signin.html')
    if (request.method == 'POST'):
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
