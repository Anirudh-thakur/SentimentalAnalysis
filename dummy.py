def get_feature_dictionary(corpus):
    print(corpus)
    result = dict()
    for i,ele in enumerate(corpus):
        print(ele[0])
        for word in ele[0]:
            if word not in result.keys():
                result[word] = i
    return result

print(get_feature_dictionary((['I','am','Anirudh'],0)))