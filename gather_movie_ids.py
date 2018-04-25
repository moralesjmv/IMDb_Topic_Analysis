
def search_movie_ids(title_array, in_str, substr):
	"""
	Get all movie IDs containing the substring.
	"""
	lines = in_str.splitlines()
	for i in range(len(lines)-1):
		line = lines[i]
		next_line = lines[i+1]
		if substr in line:
			title = next_line[1:-4]
			title_array.append(title)
			print(title)
	return title_array

def gather_top_movies(substr, dest_file_top):
    lines = []
    file1 = open("./top1.html", 'r').read()
    file2 = open("./top2.html", 'r').read()
    file3 = open("./top3.html", 'r').read()
    file4 = open("./top4.html", 'r').read()
    file5 = open("./top5.html", 'r').read()
    lines = search_movie_ids(lines, file1, substr)
    lines = search_movie_ids(lines, file2, substr)
    lines = search_movie_ids(lines, file3, substr)
    lines = search_movie_ids(lines, file4, substr)
    lines = search_movie_ids(lines, file5, substr)
    joined_lines = '\n'.join(lines)
    open(dest_file_top, 'w').write(joined_lines)

def gather_bottom_movies(substr, dest_file_bottom):
    lines = []
    file1 = open("./bottom1.html", 'r').read()
    file2 = open("./bottom2.html", 'r').read()
    file3 = open("./bottom3.html", 'r').read()
    file4 = open("./bottom4.html", 'r').read()
    file5 = open("./bottom5.html", 'r').read()
    lines = search_movie_ids(lines, file1, substr)
    lines = search_movie_ids(lines, file2, substr)
    lines = search_movie_ids(lines, file3, substr)
    lines = search_movie_ids(lines, file4, substr)
    lines = search_movie_ids(lines, file5, substr)
    joined_lines = '\n'.join(lines)
    open(dest_file_bottom, 'w').write(joined_lines)

def copy_movie_IDs_to(substr, dest_file_top, dest_file_bottom):
    """
    Copies movie ID's from IMDb lists to a destination file if they
    contain the given substring.
    """
    gather_top_movies(substr, dest_file_top)
    gather_bottom_movies(substr, dest_file_bottom)


#copy_movie_IDs_to('<span class="rating-cancel "><a href="/title/tt', "./movie_ids_top_250.txt", "./movie_ids_bottom_250.txt")
copy_movie_IDs_to('?ref_=adv_li_tt', "./movie_titles_top_250.txt", "./movie_titles_bottom_250.txt")
