"""
Description : Titanic
"""

## IMPORTANT: Use only the provided packages!

## SOME SYNTAX HERE.   
## I will use the "@" symbols to refer to some variables and functions. 
## For example, for the 3 lines of code below
## x = 2
## y = x * 2 
## f(y)
## I will use @x and @y to refer to variable x and y, and @f to refer to function f

import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


######################################################################
# classes
######################################################################

class Classifier(object) :

    ## THIS IS SOME GENERIC CLASS, YOU DON'T NEED TO DO ANYTHING HERE. 

    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) : ## INHERITS FROM THE @CLASSIFIER

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        # n,d = X.shape ## get number of sample and dimension
        y = [self.prediction_] * X.shape[0]
        return y

class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- an array specifying probability to survive vs. not 
        """
        self.probabilities_ = None ## should have length 2 once you call @fit

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        # in simpler wordings, find the probability of survival vs. not
        total_people = X.shape[0]
        if Counter(y).most_common(1)[0][0] == 1:
            survival = Counter(y).most_common(1)[0][1]
            #death = Counter(y).most_common(2)[1][1]
        else:
            #death = Counter(y).most_common(1)[0][1]
            survival = Counter(y).most_common(2)[1][1]
        self.probabilities_ = [1-survival/total_people, survival/total_people]
        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (check the arguments of np.random.choice) to randomly pick a value based on the given probability array @self.probabilities_

        y = None
        y = np.random.choice(2, X.shape[0], p=self.probabilities_)
        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use @train_test_split to split the data into train/test set 
    # xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, random_state = i)
    # now you can call the @clf.fit (xtrain, ytrain) and then do prediction

    train_error = 0 ## average error over all the @ntrials
    test_error = 0
    train_scores = []; test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 
    for i in range(ntrials):
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size,random_state=i)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_scores.append(metrics.accuracy_score(y_train, y_pred_train, normalize=True))
        test_scores.append(metrics.accuracy_score(y_test, y_pred_test, normalize=True))

    train_error = sum([1-x for x in train_scores])/ntrials
    test_error = sum([1-x for x in test_scores])/ntrials
    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    """for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
    """

    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    # call the function @DecisionTreeClassifier
    clf = DecisionTreeClassifier('entropy')
    clf.fit(X, y)
    train_error = 1- clf.score(X, y)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    # call the function @KNeighborsClassifier
    clf = KNeighborsClassifier(3)
    clf.fit(X, y)
    train_error = 1 - clf.score(X, y)
    print('\t-- training error: %.3f (k=3)' % train_error)
    clf = KNeighborsClassifier(5)
    clf.fit(X, y)
    train_error = 1 - clf.score(X, y)
    print('\t-- training error: %.3f (k=5)' % train_error)
    clf = KNeighborsClassifier(7)
    clf.fit(X, y)
    train_error = 1 - clf.score(X, y)
    print('\t-- training error: %.3f (k=7)' % train_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # call your function @error
    train_error, test_error = error(MajorityVoteClassifier(), X, y)
    print('\tFor Majority Vote Classifier')
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    train_error, test_error = error(RandomClassifier(), X, y)
    print('\tFor Random Classifier')
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    train_error, test_error = error(DecisionTreeClassifier('entropy'), X, y)
    print('\tFor Decision Tree Classifier')
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    train_error, test_error = error(KNeighborsClassifier(5), X, y)
    print('\tFor KNeighbors Classifier')
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    # hint: use the function @cross_val_score
    k = list(range(1,50,2))
    cv_score = [] ## track accuracy for each value of $k, should have length 25 once you're done
    for i in k:
        score = cross_val_score(KNeighborsClassifier(i), X, y, cv=10)
        cv_score.append(sum(score)/10)
    error_f = [1-x for x in cv_score]
    plt.plot(k, error_f , marker='o')
    plt.xlabel('k')
    plt.ylabel('Validation Error')
    plt.title('Validation Error vs k')
    plt.show()
    print('\t-- the best k is: %d' % (1+2*error_f.index(min(error_f))))
    print('\t-- its error is: %.3f' % (min(error_f)))
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    k = list(range(1,20))
    train_errors = []
    test_errors = []

    for i in k:
       train_error, test_error = error(DecisionTreeClassifier('entropy', max_depth=i), X, y)
       train_errors.append(train_error)
       test_errors.append(test_error)

    plt.plot(k, train_errors, marker='o', label='train_errors')
    plt.plot(k, test_errors, marker='o', label='test_errors')
    plt.legend(loc ='lower left')
    plt.xlabel('max_depth')
    plt.ylabel('Error')
    plt.title('Error vs max-depth')
    plt.show()
    print('\t-- the best depth is: %d' % (1+test_errors.index(min(test_errors))))
    print('\t-- its error is: %.3f' % (min(test_errors)))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_split, X_test, y_split, y_test=train_test_split(X, y, test_size=0.1, random_state=1234)
    DT_train_errors = []
    DT_test_errors = []
    kNN_train_errors = []
    kNN_test_errors = []
    ntrials = 100
    clf = DecisionTreeClassifier('entropy', max_depth=6)
    k = list(range(1,10))
    for i in k:
        train_error = 0
        test_error = 0
        train_scores = []
        test_scores = []
        for j in range(ntrials):
            X_train, X_noneed, y_train, y_noneed=train_test_split(X_split, y_split, train_size=(i/10), random_state=j)
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            train_scores.append(metrics.accuracy_score(y_train, y_pred_train, normalize=True))
            test_scores.append(metrics.accuracy_score(y_test, y_pred_test, normalize=True))

        train_error = sum([1-x for x in train_scores])/ntrials
        test_error = sum([1-x for x in test_scores])/ntrials
        DT_train_errors.append(train_error)
        DT_test_errors.append(test_error)
    k=[x/10 for x in k]
    plt.plot(k, DT_train_errors, marker='o', label='DT_train_errors')
    plt.plot(k, DT_test_errors, marker='o', label='DT_test_errors')
    #plt.legend(loc ='lower left')
    plt.xlabel('Training Data Proportion')
    plt.ylabel('Error')
    #plt.title('Decision Tree Learning Curve')
    #plt.show()


    clf = KNeighborsClassifier(7)
    k = list(range(1,10))
    for i in k:
        train_error = 0
        test_error = 0
        train_scores = []
        test_scores = []
        for j in range(ntrials):
            X_train, X_noneed, y_train, y_noneed=train_test_split(X, y, train_size=(i/10),random_state=j)
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            train_scores.append(metrics.accuracy_score(y_train, y_pred_train, normalize=True))
            test_scores.append(metrics.accuracy_score(y_test, y_pred_test, normalize=True))

        train_error = sum([1-x for x in train_scores])/ntrials
        test_error = sum([1-x for x in test_scores])/ntrials
        kNN_train_errors.append(train_error)
        kNN_test_errors.append(test_error)
    k=[x/10 for x in k]
    plt.plot(k, kNN_train_errors, marker='o', label='kNN_train_errors')
    plt.plot(k, kNN_test_errors, marker='o', label='kNN_test_errors')
    plt.legend(loc ='lower left')
    #plt.xlabel('Training Data Proportion')
    #plt.ylabel('Error')
    plt.title('Learning Curve')
    plt.show()
    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
