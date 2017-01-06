from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from collections import Counter
import numpy as np


class PermutationImportance(object):

    def __init__(self):
        pass

    def featureImportances(self, rf, X, y, nIters=1):
        '''
        Given a trained random forest instance, the training data, and labels
        calculate the feature importances. Currently in scikit-learn the
        feature importance is calculated as weighted information gain
        associated with each feature across all trees. This class calculates
        the feature importance as the decrease in out-of-bag score for each
        feature, when that feature is randomly permuted. I.e. if we randomly
        scramble a feature's values how much worse do our out of bag
        predictions become?

        Inputs:
        rf - a trained instance of sklearn's either RandomForestClassifier or
            RandomForestRegressor
        X - numpy array - the data used to train the random forest instance
        y - numpy array - the labels used to train the random forest instance
        nIters - integer - the number of times to scramble a feature and
            calculate the out-of-bag score. Increasing nIters will increase
            run-time but decrease variance in results.

        Outputs:
        featureImportances - numpy array - the change in out-of-bag score
            associated with each feature, when that feature is scrambled
        '''

        self.rf = rf
        self.X = X.copy()
        self.y = y

        nSamples, nFeatures = self.X.shape
        allInd = np.arange(0, nSamples)

        # get the oobIndices
        unsampledIndices = self._getOOBIndices()

        oobScoreScrambled = np.zeros(X.shape[1])

        if not rf.oob_score:
            oobScore = self._calcOOBScore(unsampledIndices)
        else:
            oobScore = rf.oob_score_

        # loop over features:
        for i in xrange(nFeatures):
            scores = []

            for j in xrange(nIters):
                # #### scramble column and overwrite in initial training array
                scrambleInd = np.random.permutation(allInd)
                self.X[:, i] = self.X[:, i][scrambleInd]

                # #### calculate the new oob score and store in the numpy array
                scores.append(self._calcOOBScore(unsampledIndices))

                # #### set column back to normal
                unscrambleInd = np.argsort(scrambleInd)
                self.X[:, i] = self.X[:, i][unscrambleInd]

            oobScoreScrambled[i] = np.mean(scores)

        # take difference between base oob score and the score for each
        # scrambled feature
        featureImportances = np.apply_along_axis(lambda x: oobScore - x,
                                                 0, oobScoreScrambled)

        return featureImportances

    def _calcOOBScore(self, oobInd):
        '''
        Calculate the out of bag score, given a trained instance of a
        RandomForestClassifier (from sklearn), the training data, the labels,
        and the indices of the unsampled points for each tree in the random
        forest.

        Inputs:
        rf - sklearn RandomForestClassifier instance, fit to data
        X - training data (n, k) shape with n = number of samples
            k = number of features
        y - training labels (n,) shape
        oobInd - dictionary with integer keys corresponding to each tree in the
            random forest, and values as numpy arrays of the unsampled indices
            for each tree

        Output:
            float - the random forest's out-of-bag accuracy
        '''

        oobForestPreds = {}

        if type(self.rf) is RandomForestClassifier:
            for i, tree in enumerate(self.rf.estimators_):
                # get predictions on out of bag indices for each tree
                oobTreePreds = tree.predict(self.X[oobInd[i], :])

                # create a dictionary entry for each index in the original
                # dataset. append the tree predictions to a Counter matching
                # each entry
                for j in xrange(len(oobInd[i])):
                    ind = oobInd[i][j]

                    if ind not in oobForestPreds:
                        oobForestPreds[ind] = Counter()

                    oobForestPreds[ind].update([oobTreePreds[j]])
        elif type(self.rf) is RandomForestRegressor:
            for i, tree in enumerate(self.rf.estimators_):
                # get predictions on out of bag indices for each tree
                oobTreePreds = tree.predict(self.X[oobInd[i], :])

                # create a dictionary entry for each index in the original
                # dataset. append the tree predictions to a list matching each
                # entry
                for j in xrange(len(oobInd[i])):
                    ind = oobInd[i][j]

                    if ind not in oobForestPreds:
                        oobForestPreds[ind] = []

                    oobForestPreds[ind].append(oobTreePreds[j])
        else:
            # throw error, rf is not the right class
            raise TypeError(
                'rf is not an sklearn random forest class instance')

        # subset the original labels by the final out-of-bag indices, incase
        # some points were not included
        oobIndices = np.array(oobForestPreds.keys())
        yOob = self.y[oobIndices]

        ensemblePreds = np.zeros(len(oobIndices))

        if type(self.rf) is RandomForestClassifier:
            # get the class prediction for each oob index
            for i in xrange(len(oobIndices)):
                ensemblePreds[i] = oobForestPreds[i].most_common(1)[0][0]

            # calculate the out of bag accuracy
            return accuracy_score(yOob, ensemblePreds)
        elif type(self.rf) is RandomForestRegressor:
            # get the value prediction for each oob index
            for i in xrange(len(oobIndices)):
                ensemblePreds[i] = np.mean(oobForestPreds[i])

            # calculate the out of bag MSE
            return r2_score(yOob, ensemblePreds)
        else:
            return None

    def _getOOBIndices(self):
        '''
        Retrieve the indices of the points that were not sampled for each
        tree's bootstrap sample.

        Inputs:
        X as training data, rf as instance of sk-learn RandomForestClassifier
        class

        Output:
        unsampledIndices - dictionary with keys as integers corresponding to
            each tree and values as numpy arrays of the unsampled points for
            each tree
        '''
        nSamples = self.X.shape[0]

        unsampledIndices = {}

        for i, tree in enumerate(self.rf.estimators_):

            # Here at each iteration we obtain out of bag samples for every
            # tree.
            unsampledIndices[i] = _generate_unsampled_indices(
                tree.random_state, nSamples)

        return unsampledIndices


if __name__ == "__main__":
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    rfC = RandomForestClassifier()
    rfC.n_estimators = 100
    rfC.oob_score = True
    rfC.fit(X, y)

    print "#######\n--Classification on Iris DataSet--\n#######"
    oobC = PermutationImportance()
    print "Weighted Avg Information Gain feature importances:"
    print rfC.feature_importances_
    print "Permutation importances:"
    print oobC.featureImportances(rfC, X, y, 5)

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    rfR = RandomForestRegressor()
    rfR.n_estimators = 100
    rfR.oob_score = True
    rfR.fit(X, y)

    print "\n"
    print "#######\n--Regression on Boston DataSet--\n#######"
    oobR = PermutationImportance()
    print "Weighted Avg Information Gain feature importances:"
    print rfR.feature_importances_
    print "Permutation importances:"
    print oobR.featureImportances(rfR, X, y, 5)
