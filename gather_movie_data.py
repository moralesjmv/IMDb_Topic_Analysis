import imdb

def store_movie_data(dest):
    """
    Stores movie titles and summaries into a file since doing IMDb calls take too long
    """

    top_movies = open("./movie_ids_top_250.txt", 'r').read()
    bottom_movies = open("./movie_ids_bottom_250.txt", 'r').read()

    ia = imdb.IMDb()
    summaries = []
    n = 0
    for line in bottom_movies.strip().split('\n'):
        print("Progress: " + str(n) + "%")
        n += 1
        movie = ia.get_movie(line)
        summaries.append(str(movie['plot']))
        #print(movie['plot'])
    joined_lines = '\n'.join(summaries)
    open(dest,'w').write(joined_lines)

top_summaries = []
bottom_summaries = []
store_movie_data("./movie_summaries_bottom_250.txt")
#load_data_into_array(top_summaries, bottom_summaries)
