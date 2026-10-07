import numpy as np

from nn.model import Array

class StandardScaler:
    """
    standardize features by removing mean and scaling to unit variance.
    This method is commonly used in machine learning to normalize the features of a dataset.
    It is particularly useful when the features have different units or scales, as it ensures that each feature contributes equally to the analysis.
    This scaler is also useful when the data is not normally distributed, as it can help to make the data more normally distributed.
    
    Equation:
        z = (x - mean) / std
    
    Attributes:
        copy (bool): if True, copy input data before transformation
        with_mean (bool): if True, center the data before scaling
        with_std (bool): if True, scale the data to unit variance
        mean_ (array): the mean value for each feature in the training set
        scale_ (array): per feature standard deviation
    """


    def __init__(self, copy: bool = True, with_mean: bool = True, with_std: bool = True):
        """initialize StandardScaler
        
        Args:
            copy (bool, optional): if True, copy input data. default is True
            with_mean (bool, optional): if True, center the data. default is True
            with_std (bool, optional): if True, scale to unit variance. default is True
        """
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Array | None = None
        self.scale_: Array | None = None

    def fit(self, X: Array):
        """compute mean and standard deviation for later scaling
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            StandardScaler: returns self for method chaining
        """
        if not self.copy and X.dtype != float:
            X[:] = X.astype(float)
        elif self.copy:
            X = X.astype(float)

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = None

        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            if self.scale_ is None:
                raise ValueError("At least one feature has zero standard deviation, cannot scale to unit variance.")
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = None

        return self

    def transform(self, X: Array):
        """perform standardization by centering and scaling
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            array: transformed feature matrix of same shape as X
        """
        X = np.array(X)
        if self.copy:
            X = X.copy()

        if not self.copy and X.dtype != float:
            X[:] = X.astype(float)
        elif self.copy:
            X = X.astype(float)

        if self.with_mean and self.mean_ is not None:
            X -= self.mean_
        if self.with_std and self.scale_ is not None:
            X /= self.scale_

        return X

    def fit_transform(self, X: Array):
        """fit to data, then transform it
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            array: transformed feature matrix of same shape as X
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, transformed_X: Array):
        """scale back the data to the original representation
        
        Args:
            transformed_X (array-like): transformed feature matrix of shape
                (n_samples, n_features)
        
        Returns:
            array: original feature matrix
        """
        if self.copy:
            transformed_X = transformed_X.copy()

        transformed_X = transformed_X.astype(float)

        if self.with_std and self.scale_ is not None:
            transformed_X *= self.scale_
        if self.with_mean and self.mean_ is not None:
            transformed_X += self.mean_

        return transformed_X
    