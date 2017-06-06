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
    # TODO: Continue working from the loaded model, because there is a lot of future with Python

    raise NotImplementedError()


if __name__ == '__main__':
    main()
