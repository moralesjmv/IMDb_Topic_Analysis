import imdb

def get_movie_data():
    top_movies = open("./movie_ids_top_250.txt", 'r').read()
    ia = imdb.IMDb()
    for line in top_movies.strip().split('\n'):
        movie = ia.get_movie(line)
        print(movie['title'])
        print(movie['plot'])

get_movie_data()
