### PermutationImportance Class

This class is implements the permutation importance metric for random forests, which is not currently (as of 1/6/17) implemented in scikit-learn's random forest implementations.

Currently in scikit-learn the feature importance is calculated as weighted information gain associated with each feature across all trees. This class calculates the feature importance as the change in out-of-bag score for each feature, when that feature is randomly permuted. I.e. if we randomly scramble a feature's values how much worse (or better!) do our out of bag predictions become?

For classification the base scikit-learn ".oob_score_" metric is accuracy and for regression the base score is R^2. This class returns the difference between the base ".oob_score_" and the out-of-bag score calculated after scrambling each column. The higher the difference, the more important a feature was and conversely the lower the score the less important a feature was.

Install with:
```
pip install rf_perm_feat_import
```

Example Usage:
```python
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

iris = datasets.load_iris()
X = iris.data
y = iris.target

rfC = RandomForestClassifier(n_estimators=100, oob_score=True)
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

rfR = RandomForestRegressor(n_estimators=100, oob_score=True)
rfR.fit(X, y)

print "\n"
print "#######\n--Regression on Boston DataSet--\n#######"
oobR = PermutationImportance()
print "Weighted Avg Information Gain feature importances:"
print rfR.feature_importances_
print "Permutation importances:"
print oobR.featureImportances(rfR, X, y, 5)
```
