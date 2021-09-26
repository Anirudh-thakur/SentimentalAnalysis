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
#print(X)


def normalize(X):
    for i, features in enumerate(X):
        maximum = np.amax(features)
        minimum = np.amin(features)
        #print("Maximum:{}".format(maximum))
        #print("Minimum:{}".format(minimum))
        for j, ele in enumerate(features):
            if minimum == maximum:
                X[i][j] = 0
            else:
                X[i][j] = abs((X[i][j] - minimum))/abs((maximum-minimum))
            #print(newX[i][j])
            #print(X[i][j])
    return X


print(normalize(X))
print(normalize(np.ones(shape=(3, 4))))


def evaluate_predictions(Y_pred, Y_test):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(Y_test)):
        if Y_pred[i] == Y_test[i] and Y_pred[i] == 1:
            tp += 1
        elif Y_test[i] == 0 and Y_pred[i] == 1:
            fp += 1
        elif Y_test[i] == 1 and Y_pred[i] == 0:
            fn += 1
        else:
            pass
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fmeasure = 2 * ((precision*recall)/(precision+recall))
    return (precision, recall, fmeasure)


#print(evaluate_predictions(np.ones(10), np.ones(10)))
