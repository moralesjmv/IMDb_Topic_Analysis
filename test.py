import matplotlib.pyplot as plt, mpld3
def plot_image():
    fig = plt.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)
    mpld3.show()
    return mpld3.save_html(fig, "test.html")

plot_image()
