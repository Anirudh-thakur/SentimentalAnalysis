import re
import sys

import nltk
import numpy
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    file = open(corpus_path)
    text = file.read()
    paragraph = text.split("\n")
    result = []
    for para in paragraph:
        if len(para) != 0:
            snippet = para.split(" ")
            temp = snippet[-1]
            snippet = snippet[:-1]
            if "." in str(temp):
                snippet.append('.')
            if "1" in str(temp):
                label = 1
            else:
                label = 0
            print(snippet)
            Sentiment_tuple = (snippet,label)
        result.append(Sentiment_tuple)

    return result

    """
    sentences = sent_tokenize(str(paragraph))
    result = []
    for sentence in sentences:
        temp = sentence.split("\\t")
        print(temp)
        snippet = word_tokenize(temp[0])
        #label = int(str(temp[1]))
        print(temp[1])
        #result.append(tuple(snippet,label))
    print(result)
    """


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    pass


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    pass


# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    pass
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    pass


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    pass


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    pass


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    load_corpus(corpus_path)


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    pass


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    load_corpus(test)


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    pass


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
