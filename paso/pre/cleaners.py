# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# rom pandas.util._validators import validate_bool_kwarg
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# paso import
from paso.base import _Check_No_NA_Values
from paso.base import pasoFunction, pasoDecorators, raise_PasoError
from loguru import logger

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#########0-0-0


class Transform_Values_Ratios_to_Missing(pasoFunction):
    """

        Different values can indicate, a value is missing. For example,
        - ``999`` could mean "did not answer" in some features.
        - ``NA`` could mean not-applicable for this feature/record.
        - ``-1`` could mean missing for this feature/record-1could mean missing for this feature/record`.
        and so on.

        Note:
            Detecting and correcting for missing and outlier (good or bad)
            values is an evolving area of research.

    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, X, **kwargs):
        """
            Parameters:
                Xarg: (DataFrame)
                missing_values: (list) default []
            Returns:
                    dataframe
            Raises:
            Note:
                So far, None of paso servives are invoked.
        """
        rd = {}
        for value in self.missing_values:
            rd[value] = np.nan
        X.replace(rd, inplace=True)

        return X


########## 1-1-1
class Missing_Values_Ratios(pasoFunction):
    """
        Having a sufficiently large ratio of missing values for
        a feature renders it statistically irrelevant, you can
        remove this feature from the dataset. Similarly for a
        row with a large ratio of missing values (an observation)
        renders it statistically irrelevant,and you remove this
        row from the dataset.

        An extra row (each features missing value ratio) and an
        extra feature (each row missing value ratio) is added
        to the returned **pandas** dataframe. The missing value
        ratios are kept in the class instance attribute
        ``missing_value_ratios``.
    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, X, **kwargs):
        """
            Parameters:
                Xarg: (DataFrame)
                missing_values: (list) default []
            Returns:
                    dataframe
                    self.rows_mvr: features removed
            Raises:
            Note:
                So far, None of paso servives are invoked.
        """

        self.features_mvr = (X.shape[0] - X.count(axis=0)) / X.shape[0]
        self.rows_mvr = (X.shape[1] - X.count(axis=1)) / X.shape[1]
        if self.row_rmvr:
            X["mvr"] = self.rows_mvr.values

        return X


########## 2-2-2
class Impute_Missing_Values(pasoFunction):
    """
        Having a sufficiently large ratio of missing values for
        a feature renders it statistically irrelevant, you can
        remove this feature from the dataset. Similarly for a
        row with a large ratio of missing values (an observation)
        renders it statistically irrelevant,and you remove this
        row from the dataset.

        An extra row (each features missing value ratio) and an
        extra feature (each row missing value ratio) is added
        to the returned **pandas** dataframe. The missing value
        ratios are kept in the class instance attribute
        ``missing_value_ratios``.

        Note:
            Detecting and correcting for missing and outlier (good or bad)
            values is an evolving area of research.
    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, **kwargs):
        """
            Parameters:
                Xarg: (DataFrame)
                missing_values: (list) default []
            Returns:
                    dataframe
                    self.nfeatures_missing_value: number of features with missing values
                    self.nrows_missing_value: number of features with missing values
            Raises:
            Note:
                So far, None of paso servives are invoked.
        """
        rd = {}
        for value in self.missing_values:
            rd[value] = np.nan
        X.replace(rd, inplace=True)

        self.nrows_missing_value = X.shape[0] - X.count(axis=0)
        self.nfeatures_missind_value = X.shape[1] - X.count(axis=1)
        if self.row_rmvr:
            X["was_NA"] = self.nrows_missing_value.values

        return X


########## -3-3-3
class Dupilicate_Features_by_Values(pasoFunction):
    """
        If a feature has the same values by index as another feature
        then one of those features should be deleted. The duplicate
        feature is redundant and will have no predictive power. 

        Duplicate features are quite common as an enterprise's
        database or data lake ages and different data sources are added.
    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, X, **kwargs):
        """
            Parameters:
                Xarg: (DataFrame)
                ignore: (list) default []
                    The features (column names) not to eliminate.
                    Usually this keywod argument is used for the target feature
                    that is maybe present in the training dataset.
                inplace:
            Returns:
                X: X: (DataFrame)
                    transformed X DataFrame
            Raises:
            Note:
                If feature in
        """
        k = [f for f in X.columns]
        equal = {}

        for nth, f in enumerate(X.columns[0:-1]):
            if f in self.ignore:
                break
            # the twisted logic below is for speed. most values are ne
            for mth, g in enumerate(X.columns[nth + 1 :]):
                if (X[f].values != X[g].values).any():
                    pass
                elif g not in equal:
                    equal[g] = 1

        drop_list = list(equal.keys())
        if len(drop_list) > 0:
            X.drop(columns=drop_list, inplace=True, index=1)
            if self.verbose:
                logger.info("Duplicate_Features_Removed: {}".format(str(drop_list)))

        return X


########## 4-4-4
class Features_with_Single_Unique_Value(pasoFunction):
    """
    This class finds all the features which have only one unique value.
    The variation between values is zero. All these features are removed from
    the dataframe as they have no predictive ability.
    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, X, **kwargs):
        """
            Parameters:
                Xarg: (DataFrame)
                ignore: (list) default []
                    The features (column names) not to eliminate.
                    Usually this keywod argument is used for the target feature
                    that is maybe present in the training dataset.
                inplace:
            Returns:
                X: X: (DataFrame)
                    transformed X DataFrame
            Raises:
            Note:
        """
        efs = []
        n = 0
        #        import pdb; pdb.set_trace() # debugging starts here
        for f in X.columns:
            if f in self.ignore:
                pass
            # weirdness where df has
            # 2 or more features with same name
            #               len(df[f].squeeze().shape) == 1) and
            elif X[f].nunique() == 1:
                efs.append(f)
                n += 1
        if n > 0:
            if self.verbose:
                logger.info(
                    "Eliminated Eliminate_Single_Unique_Value_Features {}".format(
                        str(efs)
                    )
                )
            for f in efs:
                X.drop(f, inplace=True, axis=1)

        return X


########## 5-5-5
class Features_Variances(pasoFunction):
    """
    This class finds all the variance of each feature and returns a dataframe
    where the 1st column is the feature string and the 2nd column is the variance of that feature.
    This class is a diagnostic tool that is used to decide if the variance is small
    and thus will have low predictive power.  Care should be used before eliminating
    any feature and a 2nd opinion of the **SHAP** value should be used in order
    to reach a decision to remove a feature.
    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, X, **kwargs):
        """
            Parameters:
                Xarg: (DataFrame)
                ignore: (list) default []
                    The features (column names) not to eliminate.
                    Usually this keywod argument is used for the target feature
                    that is maybe present in the training dataset.
                inplace:

            Returns:
                    dataframe

            Raises:

            Note:
                So far, None of paso servives are invoked.


        """
        tmp_list = []
        for feature in X.columns:
            tmp_list.append([feature, X[feature].std() / X[feature].mean()])

        return pd.DataFrame(tmp_list, columns=["Feature", "Variance/Mean"])


########## 6-6-6
class Feature_Feature_Correlation(pasoFunction):
    """
    If any given Feature has an absolute high correlation coefficient
    with another feature (open interval -1.0,1.0) then is very likely
    the second one of them will have low predictive power as it is
    redundant with the other.

    Usually the Pearson correlation coefficient is used, which is
    sensitive only to a linear relationship between two variables
    (which may be present even when one variable is a nonlinear
    function of the other). A Pearson correlation coefficient
    of -0.97 is a strong negative correlation while a correlation
    of 0.10 would be a weak positive correlation.

    Spearman's rank correlation coefficient is a measure of how well
    the relationship between two variables can be described by a
    monotonic function. Kendall's rank correlation coefficient is
    statistic used to measure the ordinal association between two
    measured quantities. Spearman's rank correlation coefficient is
    the more widely used rank correlation coefficient. However,
    Kendall's is easier to understand.

    In most of the situations, the interpretations of Kendall and
    Spearman rank correlation coefficient are very similar to
    Pearson correlation coefficient and thus usually lead to the
    same diagnosis. The paso class calculates Peason's,or Spearman's,
    or Kendall's correlation co-efficients for all feature pairs of
    the dataset.

    One of the features of the feature-pair should be removed for
    its negligible effect on the prediction. Again, this class is
    a diagnostic that indicates if one feature of will have low
    predictive power. Care should be used before eliminating any
    feature to look at the **SHAP** value, (sd/mean) and correlation
    co-efficient in order to reach a decision to remove a feature.

    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, X, **kwargs):
        """
            Parameters:
                X: (DataFrame)
                method : {‘pearson’, ‘kendall’, ‘spearman’} or callable
                    pearson : standard correlation coefficient
                    kendall : Kendall Tau correlation coefficient
                    spearman : Spearman rank correlation
                    callable: callable with input two 1d ndarrays
                threshold :   (float)
                    keep If abs(value > threshold
                inplace : boolean

            Returns:
                    Correlation of X dataframe

            Raises:

            Note:
        """
        self.threshold = threshold
        if self.verbose:
            logger.info("Correlation method: {}", self.method)
        if self.verbose:
            logger.info("Correlation threshold: {}", self.threshold)
        corr = X.corr(method=self.method)
        for f in corr.columns:
            corr.loc[((corr[f] < self.threshold) & (corr[f] > -self.threshold)), f] = 0
        return corr

    def plot(self, X, mirror=False, xsize=10, xysize=10):
        if self.transformed:
            corr = X
            fig, ax = plt.subplots(figsize=(xsize, xysize))
            colormap = sns.diverging_palette(220, 10, as_cmap=True)
            if mirror:
                sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
                plt.xticks(range(len(corr.columns)), corr.columns)
                plt.yticks(range(len(corr.columns)), corr.columns)
            else:
                dropSelf = np.zeros_like(corr)
                dropSelf[np.triu_indices_from(dropSelf)] = True
                colormap = sns.diverging_palette(220, 10, as_cmap=True)
                sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
                plt.xticks(range(len(corr.columns)), corr.columns)
                plt.yticks(range(len(corr.columns)), corr.columns)

            plt.show()
        else:
            log.error(
                "X must be  transformed, correlation coefficents calulated, before heatmap."
            )
            raise PasoError


########## 8-8-8
class Remove_Features(pasoFunction):
    """
    This class finds all the features which have only one unique value.
    The variation between values is zero. All these features are removed from
    the dataframe as they have no predictive ability.
    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, X, **wargs):
        """
            Parameters:
                Xarg: (DataFrame)
                remove: (list) default []

                    The features (column names) to eliminate.
                inplace:

            Returns:
                X: X: (DataFrame)
                    transformed X DataFrame
            Raises:
            Note:
        """
        X.drop(self.remove, axis=1)
        return X


########## 9-9-9
class Features_not_in_train_or_test(pasoFunction):
    """
    If the train or test datasets have features the other
    does not, then those features will have no predictive
    power and should be removed from both datasets.
    The exception being the target feature that is present
    in the training dataset of a supervised problem. 
    Duplicate features are quite common as an enterprise's
    database or datalake ages and different data sources are added.
    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:

                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(_Check_No_NAs=False)
    def transform(self, X, **kwargs):
        """
            Parameters:
                X: (DataFrame)
                Y=None: (DataFrame): default None
                ignore: (list) default []
                    The features (column names) not to eliminate.
                    Usually this keywod argument is used for the target feature
                    that is maybe present in the training dataset.

            Returns:
                X: X: (DataFrame)
                    EliminateUnviableFeatures transformed train DataFrame
                Y: Y: (DataFrame)
                    EliminateUnviableFeatures transformed test DataFrame

            Raises:

        """
        if self.inplace:
            yarg = self.y
        else:
            yarg = sellf.y.copy()
        _Check_No_NA_Values(yarg)

        rem = set(X.columns).difference(set(yarg.columns))
        x_features_cleaned_efs = []
        x_features_cleaned = 0
        if len(rem) >= 1:
            for f in rem:
                if f not in self.ignore:
                    X.drop(f, inplace=True, axis=1)
                    x_features_cleaned_efs.append(f)
                    x_features_cleaned += 1
        if x_features_cleaned > 0:
            if self.verbose:
                logger.info(
                    "Clean_Featuresn_in_X: {}".format(str(x_features_cleaned_efs))
                )

        rem = set(yarg.columns).difference(set(X.columns))
        y_features_cleaned_efs = []
        y_features_cleaned = 0
        if len(rem) >= 1:
            for f in rem:
                if f not in self.ignore:
                    yarg.drop(f, inplace=True, axis=1)
                    y_features_cleaned_efs.append(f)
                    y_features_cleaned += 1
        if y_features_cleaned > 0:
            if self.verbose:
                logger.info(
                    "Clean_Features_not_in_y {}".format(str(y_features_cleaned_efs))
                )

        if len(X.columns) == 0:
            logger.error(
                "Clean_Features_not_in_X_or_test:transform:X and Y are orthogonal."
            )
            raise PasoError()

        return [X, yarg]

    def reset(self):
        super().reset()

        return self

    def write(self, filepath_x="", filepath_y=""):
        """
        Writes ``f(x) and f(y)`` to local fs if ``filepath`` given.

        Parameters:
            filepath_x: str, default: ''
                filepath where the transformer f_x dataframe should be written
                An error will result if not a valid path string.
            filepath_y: str, default: ''
                filepath where the transformer f_y dataframe should be written
                An error will result if not a valid path string.
        Returns:
           self

        Raises:
            PasoError(\
                logger.error("filepath_x and filepath_y must have non-blank filename."))

            PasoError(\"Must write f_x before read.\"  )

        """
        if self.transformed:
            if filepath_x != "" and filepath_y != "":
                self.persisted = True
                self.save_file_name = [filepath_x, filepath_y]
                self.f_x.to_parquet(filepath_x)
                self.f_x.to_parquet(filepath_y)
                return self
            else:
                logger.error("filepath_x and filepath_y must have non-blank filename.")
                raise PasoError()
        else:
            logger.error("Must predict or transform before write operation.")
            raise PasoError()

    def read(self, filepath_x="", filepath_y=""):
        """
        Reads the transform ``f(x) and f(y)`` from the ``filepath`` parameter given.
        Uses ``filepath`` previously used if passed parameter ``filepath``
        is blank.

        Parameters:
            filepath_x: str, default: ''
                filepath where the transformer f_x dataframe should be read.
                An error will result if not a valid path string.
                If ``filepath``is blank then last valid ``filepath`` will be used.

            filepath_y str, default: ''
                filepath where the transformer f_x dataframe should be read.
                An error will result if not a valid path string.
                If ``filepath``is blank then last valid ``filepath`` will be used.

        Returns:
            f(X)

            f(Y)

        Raises:
            PasoError(\"Must write f_x before read.\"  )
        """
        if filepath_x != "":
            self.save_file_name[0] = filepath_x
        if filepath_y != "":
            self.save_file_name[1] = filepath_y
        if self.persisted:
            self.f_x = pd.read_parquet(self.save_file_name[0])
            self.f_y = pd.read_parquet(self.save_file_name[1])
            return self.f_x, self.f_y
        else:
            logger.error("Must write f_x and f_y before read.")
            raise PasoError()


class Class_Balance(pasoFunction):
    """
    Currently, **paso** supports only the imbalanced-learn
    package. This package is very comprehensive with
    examples on how to transform (clean) different types of
    imbalanced class data.

    Note:

        Warning:
            Only **SMOTEC** can balance datasets with categorical features. All
            others will accept a  dataset only with continuous features.

    """

    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.over_sampling import BorderlineSMOTE, SMOTENC, SVMSMOTE

    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.under_sampling import ClusterCentroids
    from imblearn.under_sampling import (
        NearMiss,
        EditedNearestNeighbours,
        RepeatedEditedNearestNeighbours,
    )
    from imblearn.under_sampling import CondensedNearestNeighbour, OneSidedSelection

    __Class_Balancers__ = {
        "RanZOverSample": RandomOverSampler,
        "SMOTE": SMOTE,
        "ADASYN": ADASYN,
        "BorderLineSMOTE": BorderlineSMOTE,
        "SVMSMOTE": SVMSMOTE,
        "SMOTENC": SMOTENC,
        "RandomUnderSample": RandomUnderSampler,
        "ClusterCentroids": ClusterCentroids,
        "NearMiss": NearMiss,
        "EditedNearestNeighbour": EditedNearestNeighbours,
        "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours,
        "CondensedNearestNeighbour": CondensedNearestNeighbour,
        "OneSidedSelection": OneSidedSelection,
    }

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                ontological_filepath:
            Returns:
                self
        """

        super().__init__()

    def classBalancers(self):
        """
        Parameters:
            None

        Returns:
            List of available class Balancers ames.
        """
        return list(Class_Balance.__Class_Balancers__.keys())

    @pasoDecorators.TransformWrapnarg(array=False, narg=3, nreturn=1)
    def transform(self, X, **kwargs):
        """
            Parameters:
                name:
                verbose: (str)(default)
                        True, logging
                        on / off(``verbose = False``)
            Returns:
                self
        """

        if self.name in Class_Balance.__Class_Balancers__:
            self.model = Class_Balance.__Class_Balancers__[self.name](**self.kwargs)
            if self.verbose:
                logger.info("Class_Balance:: {}".format(self.name))
                logger.info(
                    "Class_Balance: {} with kwargs {}".format(self.name, self.kwargs)
                )

        else:
            raise_PasoError("No Class_Balancer named: {} found.".format(self.name))

        y = X[self.targetFeature]
        X = X[X.columns.difference([self.targetFeature])]

        Xarg, yarg = self.model.fit_resample(X, y)
        if self.inplace:
            X = pd.DataFrame(Xarg, columns=X.columns)
            X[self.targetFeature] = yarg
            return X
        else:
            Xarg[self.targetFeature] = yarg
            return Xarg


class Augment_by_Class(pasoFunction):
    """
    Currently, **paso** supports claas stutured data.

    Note:
    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):
        """
            Parameters:
                name:
                verbose: (str) (default) True, logging on/off (``verbose=False``)
                ontological_filepath:
            Returns:
                self
        """
        super().__init__()

    @pasoDecorators.TransformWrapnarg(array=False, narg=4, nreturn=1)
    def transform(self, X, **kwargs):
        """
            Argument data by ratio.
                1. the dataset is class balanced first.
                1. then ratio dataset stub is applied to first class by sampling
                1. this ratio stub is kept as it must be removed later
                1. the stub ia added and the dataset rebalanced
                1. the stub is subtracted
                1. nd the dataset is rebalanced.

                the result is a dataset augmented by ratio artifial dat for each class,

            Parameters:
                X: (DataFrame)
                targetFeature: (string)
                ratio: (float) must be [0.0,1.0]. setting to 1.0 means dunle size if dataser
                    size = (1 + ratio)

            Returns:
                X: X: (DataFrame)
                    transformed X DataFrame
            Raises:

            Note:
                Because of integer roundoff. the ratio increased may not be exact

        """
        if self.name in Class_Balance.__Class_Balancers__:
            self.model = Class_Balance.__Class_Balancers__[self.name](**self.kwargs)
            if self.verbose:
                logger.info("Augment_by_Class:: {}".format(self.name))
                logger.info(
                    "Augment_by_Class: {} with kwargs {}".format(self.name, self.kwargs)
                )
        else:
            raisePasoError(
                "No Augment_by_Class/Class_Balancer named: {} found.".format(self.name)
            )
        # 1
        #        X = self.model.transform(X, targetFeature, **kwargs)
        # 2
        if self.ratio > 1.0 or self.ratio < 0.0:
            raise_PasoError(
                "Augment_by_Class: {} ratio must be [0.0,1.0], was: {}", format(ratio)
            )
        classes = X[self.targetFeature].unique()
        n_class = X[self.targetFeature].nunique()
        # 3
        stub_size = int(X.shape[0] * self.ratio / n_class)
        lowest_class = np.min(classes)
        stub = X[X[targetFeature] == lowest_class].sample(stub_size)
        # 4
        X = X.append(stub)
        X = Class_Balance(ontological_filepath=self.ontological_filepath).transform(
            X, targetFeature=self.targetFeature, verbose=self.verbose
        )
        # 5the stub is subtracted
        X.drop(X.index[stub.index], inplace=True)
        # 6 and the dataset is rebalanced.
        X = Class_Balance(ontological_filepath=self.ontological_filepath).transform(
            X, targetFeature=self.targetFeature, verbose=self.verbose
        )

        return X
