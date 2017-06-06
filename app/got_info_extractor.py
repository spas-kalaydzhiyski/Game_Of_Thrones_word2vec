# Use when writing in Python2 to allow compatibility with Python3/
# from __future__ import absolute_import, division, print_function

# used for decoding from utf-8 to unicode
import codecs
# regex
import glob
# concurrency
import multiprocessing
# used for os operations
import os

# regular expressions
import re
# natural language toolkit
import nltk
# word 2 vec representations
import gensim.models.word2vec as w2v


# --------------------------- Helper methods ----------------------------------------

def download_nltk_tools():
    nltk.download('punkt')  # pre-trained tokenizer
    nltk.download('stopwords')

# ------------------------------------------------------------------------------------


def generate_corpus():
    corpus_raw_data = u''

    with codecs.open('./books/corpus_data', 'r', 'utf-8') as cd:
        corpus_raw_data += cd.read()

    print('Corpus data loaded into memory.')
    return corpus_raw_data


# ------------------------------------------------------------------------------------

# gets only the words from a sentence, removing any extra characters
def sentence_to_wordlist(raw):
    clean = re.sub('[^A-Za-z]', ' ', raw)
    words = clean.split()
    return words

# --------------------------- Helper methods ----------------------------------------


def train(sentences):
    num_features = 100                          # more dimensions = more accuracy, more computational power required
    min_word_count = 3                          # threshold
    num_workers = multiprocessing.cpu_count()   # number of parallel threads
    context_window_size = 7                     # context-window size
    down_sample = 1e-3                          # Down-sample for frequently found words
    seed = 1                                    # seed for the random number generator, to gen reproducible results

    got_corpus2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        min_count=min_word_count,
        window=context_window_size,
        sample=down_sample,
    )

    got_corpus2vec.build_vocab(sentences)
    vocab = got_corpus2vec.wv.vocab # all the words used in the corpus

    num_examples = got_corpus2vec.corpus_count
    num_epochs = got_corpus2vec.iter
    got_corpus2vec.train(sentences, total_examples=num_examples, epochs=num_epochs)

    if not os.path.exists('./trained_model'):
        os.makedirs('trained_model')

    # save the model's state for future use
    got_corpus2vec.save(os.path.join('trained_model', 'got_model.w2v'))


def main():
    corpus = generate_corpus()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(corpus)

    sentences = []
    for raw_sent in raw_sentences:
        if raw_sent:
            sentences.append(sentence_to_wordlist(raw_sent))

    train(sentences)

if __name__ == '__main__':
    main()













































