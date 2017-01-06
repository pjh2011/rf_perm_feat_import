### PermutationImportance Class

This class is implements the permutation importance metric for random forests, which is not currently (as of 1/6/17) implemented in scikit-learn's random forest implementations.

Currently in scikit-learn the feature importance is calculated as weighted information gain associated with each feature across all trees. This class calculates the feature importance as the change in out-of-bag score for each feature, when that feature is randomly permuted. I.e. if we randomly scramble a feature's values how much worse (or better!) do our out of bag predictions become?

For classification the base scikit-learn ".oob_score_" metric is accuracy and for regression the base score is R^2. This class returns the difference between the base ".oob_score_" and the out-of-bag score calculated after scrambling each column. The higher the difference, the more important a feature was and conversely the lower the score the less important a feature was.

Usage:
```python
s = "Python syntax highlighting"
print s
```
