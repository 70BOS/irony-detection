# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:35:20 2017

@author: dell
"""

from __future__ import division
from collections import defaultdict

import random
import matplotlib.pyplot as plt

POS_LABEL = 'pos'
NEG_LABEL = 'neg'


###################################

def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_add(vec1,vec2):
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] += vec2[k]
    return dict(out)

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<len(d2) else d2  
    total = 0
    for key in smaller.iterkeys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """

    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return bow

def construct_dataset():
    """
    Build a dataset. If train is true, build training set otherwise build test set.
    The training set is a list of training examples. A training example is a tuple
    containing a dictionary (that represents features of the training example) and
    a label for that training instance (either 1 or -1).
    """
    dataset = []
    print "[constructing dataset...]"
    with open("dataset.txt") as f_in:
        for line in f_in:
            line_split=line.lower().split("\t")
            label = 1
            if line_split[1] == '0':
                label = -1
            bow = tokenize_doc(line_split[2])
            dataset.append((bow,label))
    random.shuffle(dataset)
    train = dataset[:3067]
    test = dataset[3067:]
    print "[dataset constructed.]"
    return train,test

###############################

def features_for_label(bow, label):
    """
    The full f(x,y) function. Pass in the bag-of-words for one document
    (represented as a dictionary) and a string label. Returns one big feature
    vector.  The space of keys for this feature vector is used for all classes.
    But for any single call to this function, half of all the features have to be zero.
    """
    feat_vec = defaultdict(float)
    for word, value in bow.iteritems():
        feat_vec["%s_%s" % (label, word)] = value
    return feat_vec

def predict_multiclass(bow, weights):
    """
    Takes a set of features represented as a dictionary and a weight vector that contains
    weights for features of each label (represented as a dictionary) and
    performs perceptron multi-class classification (i.e., finds the class with the highest
    score under the model.

    You may find it peculiar that the name of this function has to do with multiclass
    classification or that we are making predictions in this relatively complicated way
    given the binary classification setting. You are correct; this is a bit weird. But,
    this code looks a lot like the code for multiclass perceptron and the structured
    perceptron (i.e., making the next part of this homework easier).
    """
    pos_feat_vec = features_for_label(bow, "1")
    neg_feat_vec = features_for_label(bow, "-1")
    scores = { 1: dict_dotprod(pos_feat_vec, weights),
               -1: dict_dotprod(neg_feat_vec, weights) }
    return dict_argmax(scores)

def train(examples, stepsize, numpasses=10, do_averaging=False, devdata=None):
    """
    Trains a perceptron.
      examples: list of (featvec, label) pairs; featvec is a dictionary and label is a string
      stepsize: hyperparameter; this affects how the weights are updated
      numpasses: the number of times to loop through the dataset during training
      do_averaging: boolean that determines whether to use averaged perceptron
      devdata: the test set of examples

    returns a dictionary containing the vanilla perceptron accuracy on the train set and test set
    and the averaged perceptron's accuracy on the test set.
    """

    weights = defaultdict(float)
    S = defaultdict(float)
    
    train_acc = []
    test_acc = []
    avg_test_acc = []

    def get_averaged_weights():
        for key in S:
            S[key] /= pass_iteration+1
        theta_bar = dict_subtract(weights,S)
        for key in S:
            S[key] *= pass_iteration+1
        return theta_bar

    print "[training...stepsize = %f]" % stepsize
    
    for pass_iteration in range(numpasses):
        print "\tTraining iteration %d" % pass_iteration
        random.shuffle(examples)
        for bow, goldlabel in examples:
            
            predlabel = predict_multiclass(bow, weights)
            y = features_for_label(bow,goldlabel)
            y_prime = features_for_label(bow,predlabel)
            g = dict_subtract(y,y_prime)
            for key in g:
                g[key] *= stepsize
            if predlabel != goldlabel:
                weights = dict_add(weights,g) 
            
            for key in g:
                g[key] *= pass_iteration
            S = dict_add(S,g)

        if devdata:
            print "DEV RAW EVAL:",
            test_acc.append(do_evaluation(devdata, weights))

        if devdata and do_averaging:
            print "DEV AVG EVAL:",
            avg_test_acc.append(do_evaluation(devdata, get_averaged_weights()))

    print "[learned weights for %d features from %d examples.]" % (len(weights), len(examples))

    return { 'train_acc': train_acc,
             'test_acc': test_acc,
             'avg_test_acc': avg_test_acc,
             'weights': weights if not do_averaging else get_averaged_weights() }

def do_evaluation(examples, weights):
    """
    Compute the accuracy of a trained perceptron.
    """
    num_correct, num_total = 0, 0
    for feats, goldlabel in examples:
        predlabel = predict_multiclass(feats, weights)
        if predlabel == goldlabel:
            num_correct += 1.0
        num_total += 1.0
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

if __name__=='__main__':
    import numpy
    training_set,test_set = construct_dataset()
    inds = numpy.arange(10)
    
    sol_dict = train(training_set,stepsize=0.01, do_averaging=True, devdata=test_set)

    plt.figure(figsize=(10,20)) 
    p1=plt.subplot(4, 2, 1)
    p1.scatter(inds, sol_dict['test_acc'], c='r', marker = "o",linewidth=1)
    p1.set_title("perceptron,stepsize = 0.01")
    p1.set_xlabel("iteration") 
    p1.set_ylabel("test accuracy")
    p2=plt.subplot(4, 2, 2)
    p2.scatter(inds, sol_dict['avg_test_acc'], c='r', marker = "o",linewidth=1)
    p2.set_title("averaged perceptron,stepsize = 0.01")
    p2.set_xlabel("iteration") 
    p2.set_ylabel("test accuracy")

    
    sol_dict = train(training_set,stepsize=0.1, do_averaging=True, devdata=test_set)
    
    p3=plt.subplot(4, 2, 3)
    p3.scatter(inds, sol_dict['test_acc'], c='r', marker = "o",linewidth=1)
    p3.set_title("perceptron,stepsize = 0.1")
    p3.set_xlabel("iteration") 
    p3.set_ylabel("test accuracy")
    p4=plt.subplot(4, 2, 4)
    p4.scatter(inds, sol_dict['avg_test_acc'], c='r', marker = "o",linewidth=1)
    p4.set_title("averaged perceptron,stepsize = 0.1")
    p4.set_xlabel("iteration") 
    p4.set_ylabel("test accuracy")
    
    sol_dict = train(training_set,stepsize=1, do_averaging=True, devdata=test_set)
    
    p5=plt.subplot(4, 2, 5)
    p5.scatter(inds, sol_dict['test_acc'], c='r', marker = "o",linewidth=1)
    p5.set_title("perceptron,stepsize = 1")
    p5.set_xlabel("iteration") 
    p5.set_ylabel("test accuracy")
    p6=plt.subplot(4, 2, 6)
    p6.scatter(inds, sol_dict['avg_test_acc'], c='r', marker = "o",linewidth=1)
    p6.set_title("averaged perceptron,stepsize = 1")
    p6.set_xlabel("iteration") 
    p6.set_ylabel("test accuracy")
    
    sol_dict = train(training_set,stepsize=10, do_averaging=True, devdata=test_set)
    
    p7=plt.subplot(4, 2, 7)
    p7.scatter(inds, sol_dict['test_acc'], c='r', marker = "o",linewidth=1)
    p7.set_title("perceptron,stepsize = 10")
    p7.set_xlabel("iteration") 
    p7.set_ylabel("test accuracy")
    p8=plt.subplot(4, 2, 8)
    p8.scatter(inds, sol_dict['avg_test_acc'], c='r', marker = "o",linewidth=1)
    p8.set_title("averaged perceptron,stepsize = 10")
    p8.set_xlabel("iteration") 
    p8.set_ylabel("test accuracy")
    
    plt.savefig("perc.png")
