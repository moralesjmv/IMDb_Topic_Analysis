import imdb

def store_movie_data(dest):
    """
    Stores movie titles and summaries into a file since doing IMDb calls take too long
    """

    top_movies = open("./movie_ids_top_250.txt", 'r').read()
    bottom_movies = open("./movie_ids_bottom_250.txt", 'r').read()

    ia = imdb.IMDb()
    titles = []
    n = 0
    for line in top_movies.strip().split('\n'):
        print("Progress: " + str(n) + "%")
        n += 1
        movie = ia.get_movie(line)
        titles.append(movie['title'])
        #summaries.append(movie['plot'])
        #print(movie['title'])
        #print(movie['plot'])

    joined_lines = '\n'.join(titles)
    open(dest,'w').write(joined_lines)

def load_data_into_array(top_titles, bot_titles, top_sum, bot_sum):
    """
    Reads movie data files and stores them in arrays
    """
    return

top_titles = []
top_summaries = []
bottom_titles = []
bottom_summaries = []
store_movie_data("./movie_titles_top_250.txt")
load_data_into_array(top_titles, bottom_titles, top_sum, bottom_sum)
