### PermutationImportance

This class is implements the permutation importance metric for random forests, which is not implemented in scikit-learn's random forest implementations.

Currently in scikit-learn the feature importance is calculated as weighted information gain associated with each feature across all trees. This class calculates the feature importance as the change in out-of-bag score for each feature, when that feature is randomly permuted. I.e. if we randomly scramble a feature's values how much worse (or better!) do our out of bag predictions become?
