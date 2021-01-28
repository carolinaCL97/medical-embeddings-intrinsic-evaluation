import pandas as pd
import numpy as np
import scipy
import re

def normalizer(text, remove_tildes = True): 
"""normalizes a given string to lowercase and changes all vowels to their base form"""

    text = text.lower() #string lowering
    text = re.sub(r'[^A-Za-zñáéíóú]', ' ', text) #replaces every punctuation with a space
    if remove_tildes:
        text = re.sub('á', 'a', text) #replaces special vowels to their base forms
        text = re.sub('é', 'e', text)
        text = re.sub('í', 'i', text)
        text = re.sub('ó', 'o', text)
        text = re.sub('ú', 'u', text)
    return text


any_in = lambda a, b: any(i in b for i in a)
all_in = lambda a, b: all(i in b for i in a)

def solveAnalogy(model, k, a, b, c):
    """ Receives a word2vec gensim model, a k for the most similar words searching and 3 lists of words.
    The queried words assume the construction of the question 'a' is a 'b' as 'c' is a 'k most similar words'.
    Example:
    a = ["hombre","varón", "macho", "masculino"]
    b = ["mujer","hembra","fémina","dama"]
    c = ["rey"]
    The function returns a list of tuples in the form of (word, distance)
    """

    a_vector = np.median([model[word] for word in a], axis=0)
    b_vector = np.median([model[word] for word in b], axis=0)
    c_vector = np.median([model[word] for word in c], axis=0)

    v = b_vector - a_vector + c_vector
    most_similar = model.most_similar(np.array([v]), topn = k+len(c))
    result = [x for x in most_similar if x[0] not in c]
    return result[:k]

def evaluator(model, k, a, b, c, correct):
    responses = [response[0] for response in solveAnalogy(model, k, a, b, c)]
    return any_in(correct, responses)

def evaluate_analogy_set(model, k, analogy_set):
    """performs evaluator function for every row in the analogy_set dataframe and
    returns the number of times the analogy was solved correctly"""
    hits = 0
    for i in range(len(analogy_set)):
        result = evaluator(model, k, analogy_set['Term1'][i], analogy_set['Term2'][i], 
                 analogy_set['Term3'][i], analogy_set['Term4'][i])
        if result == True:
           hits += 1
    return hits

def pair_similarity(model, df_sim):
    """returns pearson and spearman relation between cosine similarity scores given by the model 
    and the scores set by the specialists."""

    cosine_scores = []
    for index in range(len(df_sim)):
        cosine = model.similarity(df_sim['Term1'][index], df_sim['Term2'][index])
        cosine_scores.append(cosine)   
    pearson = scipy.stats.pearsonr(cosine_scores, df_sim['Mean'])
    spearman = scipy.stats.spearmanr(cosine_scores, df_sim['Mean']) 
    return pearson[0], spearman[0]      