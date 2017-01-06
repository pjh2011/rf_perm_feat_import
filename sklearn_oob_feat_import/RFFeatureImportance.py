from sklearn.ensemble.forest import _generate_unsampled_indices
from collections import Counter
import numpy as np


class OobFeatureImportance(object):

    def __init__(self):
        pass

    def featureImportances(self, rf, X, y):
        '''
        '''

        self.rf = rf
        self.X = X
        self.y = y

        # get the oobIndices
        unsampledIndices = self.getOOBIndices_()

        oobScoreScrambled = np.zeros(X.shape[1])

        # loop over features:
        # #### scramble feature (maybe do this a few times?)
        # #### calculate the new oob score and store in the numpy array

        return oobScoreScrambled

    def calcOOBScore_(self, rf, X, y, oobInd):
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

        for i, tree in enumerate(rf.estimators_):
            # get predictions on out of bag indices for each tree
            oobTreePreds = tree.predict(X[oobInd[i], :])

            # create a dictionary entry for each index in the original
            # dataset. append the tree predictions to a Counter matching
            # each entry
            for j in xrange(len(oobInd[i])):
                ind = oobInd[i][j]

                if ind not in oobForestPreds:
                    oobForestPreds[ind] = Counter()

                oobForestPreds[ind].update([oobTreePreds[j]])

        # subset the original labels by the final out-of-bag indices, incase
        # some points were not included
        oobIndices = np.array(oobForestPreds.keys())
        yOob = y[oobIndices]

        ensemblePreds = np.zeros(len(oobIndices))

        # get the prediction for each oob index
        for i in xrange(len(oobIndices)):
            ensemblePreds[i] = oobForestPreds[i].most_common(1)[0][0]

        # calculate the out of bag accuracy
        return 1.0 * np.sum(yOob == ensemblePreds) / len(oobIndices)

    def getOOBIndices_(self):
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
