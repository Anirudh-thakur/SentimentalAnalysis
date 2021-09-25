import numpy as np
def vectorize_snippet(snippet, feature_dict):
    vector = np.zeros(len(feature_dict))
    for words in snippet:
        index = feature_dict[words]
        vector[index] += 1
    return vector

def vectorize_corpus(corpus, feature_dict):
    d = len(feature_dict.keys())
    n = len(corpus)
    X = np.zeros(shape=(n,d))
    y = np.zeros(n)
    for i,elements in enumerate(corpus):
        snippets = elements[0]
        label = elements[1]
        X[i] = vectorize_snippet(snippets, feature_dict)
        y[i] = label
    return (X,y)
X = vectorize_corpus([(['I','am','Anirudh'],0),(['I','am','Shubham'],0)],{'I':2,'am':2,'Anirudh':1,'Shubham':1})
X = X[0]
print(X)


def normalize(X):
    newX = np.zeros(shape=(len(X), len(X[0])))
    for i, features in enumerate(X):
        maximum = np.amax(features)
        minimum = np.amin(features)
        #print("Maximum:{}".format(maximum))
        #print("Minimum:{}".format(minimum))
        if maximum == minimum:
            continue
        else:
            for j, ele in enumerate(features):
                newX[i][j] = abs((X[i][j] - minimum))/abs((maximum-minimum))
                #print(newX[i][j])
                #print(X[i][j])
    return newX

print(normalize(X))
print(normalize(np.ones(shape=(3, 4))))
