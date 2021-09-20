import nltk


def is_negation(word):
    if word in negation_words:
        return True
    if word.endswith("n't"):
        return True
    return False

negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])
def tag_negation(snippet):
    TaggedWords = nltk.pos_tag(snippet)
    result = []
    negation_meta_tag = "NOT_"
    negationFlag = 0
    for i,words in enumerate(TaggedWords):
        if negationFlag == 0:
            result.append(words[0])
        else:
            result.append(negation_meta_tag+words[0])
        if is_negation(words[0]):
            if words[0] == "not" and i+1 <= len(TaggedWords)-1 and TaggedWords[i+1][0] == "only":
                pass
            else:
                negationFlag = 1
        if words[0] in negation_enders or words[0] in sentence_enders or words[1] == "JJR" or words[1] == "RBR":
                negationFlag = 0
    return result
    

print(tag_negation(['I', 'am', 'not', 'Anirudh', 'however','Thakur']))
