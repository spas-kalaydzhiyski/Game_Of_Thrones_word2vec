import gensim.models.word2vec as w2v
import os
# pretty printing in human-readable format
import pprint
# used for dimensionality reduction
import sklearn.manifold
# speedy math library (executes C code)
import numpy as np
# plotting the data
import matplotlib.pyplot as plt
# used for parsing the dataframe
import pandas as pd
# used for visualization
import seaborn as sns
# used for serialization
import pickle


def save_matrix(word_vectors_matrix):
    with open('word_vectors_matrix_2d.pickle', 'wb') as matrix_file:
        pickle._dump(word_vectors_matrix, matrix_file)


def create_and_save_DataFrame(wv_matrix_2d, got_corpus2vec):
    # init
    vocab = got_corpus2vec.wv.vocab
    points = pd.DataFrame(
        [
            (word, coordinates[0], coordinates[1])
            for word, coordinates in [
                (word, wv_matrix_2d[vocab[word].index])
                for word in vocab
            ]
        ],
        columns=['word', 'x', 'y']
    )

    # save
    with open('points.pickle', 'wb') as f:
        pickle.dump(points, f)


def plot_region(points, x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


def nearest_similarity_cosmul(got_corpus2vec, start1, end1, end2):
    similarities = got_corpus2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


def main():
    got_corpus2vec = w2v.Word2Vec.load(os.path.join('trained_model', 'got_model.w2v'))
    tSNE = sklearn.manifold.TSNE(n_components=2, random_state=0)
    #word_vectors_matrix = got_corpus2vec.wv.syn0
    # word_vectors_matrix_2d = tSNE.fit_transform(word_vectors_matrix)
    #
    # save_matrix(word_vectors_matrix_2d)

    # get matrix
    with open('word_vectors_matrix_2d.pickle', 'rb') as f:
        word_vectors_matrix_2d = pickle.load(f)

    # create_and_save_DataFrame(word_vectors_matrix_2d, got_corpus2vec)

    # get points
    points = []
    with open('points.pickle', 'rb') as f:
        points = pickle.load(f)

    sns.set_context("poster")

    # TODO: plot the coordinates

    # most_similar_list = got_corpus2vec.most_similar('Cersei')
    # for i in most_similar_list:
    #     print(i)

    # got_corpus2vec.most_similar('Jamie')
    # got_corpus2vec.most_similar('direwolf')

    # nearest_similarity_cosmul(got_corpus2vec, "Stark", "Winterfell", "Riverrun")

if __name__ == '__main__':
    main()
