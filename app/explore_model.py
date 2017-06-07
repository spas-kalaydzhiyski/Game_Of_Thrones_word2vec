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


def main():
    got_corpus2vec = w2v.Word2Vec.load(os.path.join('trained_model', 'got_model.w2v'))
    tSNE = sklearn.manifold.TSNE(n_components=2, random_state=0)
    word_vectors_matrix = got_corpus2vec.wv.syn0
    word_vectors_matrix_2d = tSNE.fit_transform(word_vectors_matrix)

    # points for plotting in 2d space
    points = pd.DataFrame(
        [
            (word, coordinates[0], coordinates[1])
            for word, coordinates in [
                (word, word_vectors_matrix_2d[got_corpus2vec.wv.vocab[word].index])
                for word in got_corpus2vec.wv.vocab
            ]
        ],
        columns=['word', 'x', 'y']
    )

    print(points.head(10))

if __name__ == '__main__':
    main()
