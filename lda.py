def setup_data_arrays(top_t, bottom_t,top_s, bottom_s):
    """
    Stores movie titles and summaries into arrays
    """

    top_movie_titles = open("./movie_titles_top_250.txt", 'r').read()
    bottom_movie_titles = open("./movie_titles_bottom_250.txt", 'r').read()
    top_movie_summaries = open("./movie_summaries_top_250.txt", 'r').read()
    bottom_movie_summaries = open("./movie_summaries_bottom_250.txt", 'r').read()

    """Collects titles"""
    for line in top_movie_titles.strip().split('\n'):
        top_t.append(line)
    for line in bottom_movie_titles.strip().split('\n'):
        bottom_t.append(line)

    """Collects summaries"""
    for line in top_movie_summaries.strip().split('\n'):
        top_s.append(line)
    for line in bottom_movie_summaries.strip().split('\n'):
        bottom_s.append(line)

top_titles = []
bottom_titles = []
top_summaries = []
bottom_summaries = []
setup_data_arrays(top_titles, bottom_titles, top_summaries, bottom_summaries)
print(top_titles)
print(top_summaries[0:5])

#TODO: LDA and K clustering following this guide
#http://brandonrose.org/clustering#Latent-Dirichlet-Allocation
