# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

from typing import Dict, List
import numpy as np
import pandas as pd

# rom pandas.util._validators import validate_bool_kwarg
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer

# paso import
from paso.base import _Check_No_NA_Values, _array_to_string
from paso.base import _dict_value, _check_non_optional_kw
from paso.base import DataFrame_to_Xy, Xy_to_DataFrame
from paso.base import pasoFunction, pasoDecorators, raise_PasoError
from paso.base import _must_be_list_tuple_int, _merge_dicts
from loguru import logger


class Imputers(pasoFunction):
    """
          Class to impute NaM from dataset. Encoding and scaling
          and other data-set preprocessing should not be done here.

          parameters:
              dataset: (DataFrame) are independent features of X

          keywords:
              kind: (list)
                  'missing': clean data substituting np.nan with how/strategy keywords.
              strategy: (list)
                  'impute': clean data substituting np.nan with how/strategy keywords.
              kind: (list)
                  'mean': clean data substituting np.nan with how/strategy keywords.

          returns:
                  X: (DataFrame)

          Note:
              Impute before other cleaning and encoding, These steps all expect
              that NaNs have been removed. Use method Cleaners.Values_to_nan()
              beforehand to change ant incatitot values to NaN.

          """

    _imputer_type_supported_ = ("numeric", "all")
    _imputer_simple_stategies_ = {
        "median": "median",
        "mean": "mean",
        "most_frequent": "most_frequent",
        "random": "random",
    }

    _imputer_advanced_stategies_ = {"knn": "knn", "mice": "mice"}

    _imputers_ = _merge_dicts(_imputer_advanced_stategies_, _imputer_simple_stategies_)

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs) -> None:
        """
        Parameters:
            description_file:

        Returns: instance of claa imputer

        """

        super().__init__()

    @staticmethod
    def imputers() -> List[str]:
        """
        Parameters:
            None

        Returns:
            List of available class inputers names.
        """
        return [k for k in Imputers._imputers_.keys()]

    @pasoDecorators.TTWrap(array=False, _Check_No_NAs=False)
    def transform(
        self,
        X: pd.DataFrame,
        verbose: bool = True,
        inplace: bool = True,
        features: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        method to transform dataset by imputing NaN values. Encoding and scaling
        and other data-set preprocessing should not be done here.

        parameters:
            X: (DataFrame) are independent features of dataset

        keywords:
            features: (list)


        returns:
                dataset: (DataFrame)
        """

        # todo pull datatypes automatically, just numericall and all??
        # todo support other data types besides i.e. most_frequent can support
        # cat/object/string if na
        # todo incorporate this A Comparison of Six Methods for Missing Data Imputation
        # https://www.omicsonline.org/open-access/a-comparison-of-six-methods-for-missing-data-imputation-2155-6180-1000224.pdf
        # https://impyute.readthedocs.io/en/master/index.html
        # todo checking passed arguments types are correct
        # enforce order of method calls

        # currently support only one learner, very brittle parser
        if self.kind == {}:
            raise_PasoError(
                "keyword kind must be present at top level:{}:".format(
                    self.ontology_kwargs
                )
            )

        if self.kind_name not in Imputers._imputers_:
            raise_PasoError(
                "no operation named: {} in imputers;: {}".format(
                    self.kind_name, Imputers._imputers_.keys()
                )
            )

        if inplace:
            y = X
        else:
            y = X.copy()

        if features == None:
            features = y.columns
            logger.warning(
                "\nKeyword arg:features: not passed. All features of dataset will be checked for imputation\n{}".format(
                    features
                )
            )
        else:
            pass

        ## some of features not in columns
        if not len(set(features).difference(set(self.Xcolumns))) == 0:
            raise_PasoError(
                "\nfeature given {} not in \ndataset columns:{}".format(
                    features, y.columns
                )
            )

        self.imputer_name = self.kind_name

        # check each attirbute for all nand
        for feature in features:
            if y[feature].nunique() <= 1:
                raise_PasoError(
                    "Impute.transform: 1 or less unique values: {}.\n Remove this feature before calling Impute.".format(
                        y[feature].unique()
                    )
                )

        if self.kind_name in Imputers._imputer_simple_stategies_:
            # assume target does not have missung values
            # todo handling target rows with missing values
            #  with simple row elimination? The idea being you should not try to predict missing values?
            imputer = SimpleImputer(strategy=self.kind_name)
            imputer.fit(y[features])
            y[features] = imputer.transform(y[features])
        elif self.kind_name in Imputers._imputer_advanced_stategies_:
            self.imputer = Imputers._imputer_advanced_stategies_[self.imputer_name](
                **self.kind_name_kwargs
            )

            self.imputer.fit(X, y)
            self.imputer_type = self.type
            y[features] = self.imputer.transform(y[features])

        return y


class Cleaners(pasoFunction):
    """
    The class of data cleaners.

    """

    _statistics = [
        "kurt",
        "mad",
        "max",
        "mean",
        "median",
        "min",
        "sem",
        "skew",
        "sum",
        "std",
        "var",
        "nunique",
        "all",
    ]

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs) -> None:
        """

        """
        super().__init__()
        self.transformed_Values_to_nan = False
        self.transformed_Delete_NA_Features = False
        self.transformed_Calculate_NA_ratio = False
        self.transformed_delete_Duplicate_Features = False
        self.transformed_Features_with_Single_Unique_Value = False
        self.transformed_Features_Statistics = False
        self.transformed_booleans = False
        self.transformed_Feature_Feature_Correlation = False

    ####### 1
    def values_to_nan(
        self,
        X: pd.DataFrame,
        values: List[str] = [],
        inplace: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Different values can indicate, a value is missing. For example,
        - ``999`` could mean "did not answer" in some features.
        - ``NA`` could mean not-applicable for this feature/record.
        - ``-1`` could mean missing for this feature/record-1could mean missing for this feature/record`.
        and so on.

        Parameters:
            X: DataFrame

            values: list or str, default []

            inplace : bool, default True
            If True, do operation inplace and return None.

            verbose: bool

        Returns: DataFrame

        """
        # todo implement c.column_missing_value_ratio
        self.values = _must_be_list_tuple_int(values)
        if self.values == []:
            return X

        y = X.replace(to_replace=values, value=np.nan, inplace=inplace)

        if verbose:
            logger.info("Values_to_nan {}".format(str(self.values)))

        self.transformed_Values_to_nan = True

        if inplace:
            return X
        else:
            return y

    ######### 2
    def delete_NA_Features(
        self,
        X: pd.DataFrame,
        threshold: float = 1.0,
        inplace: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
            Having a sufficiently large ratio of missing values for
            a feature renders it statistically irrelevant, you can
            remove this feature from the dataset.

            Parameters:
                X: DataFrame

                threshold: (float, default 1.0 . Above this theshold of a nan ratio
                 , feature is deleted.

                inplace : bool, default True
                    If True, do operation inplace and return None.

                verbose: bool


            Returns:
                    dataframe
                    self.nfeatures_missing_value: number of features with missing values
                    self.nrows_missing_value: number of features with missing values
            Raises:
        """

        self.features_above_threshold_nan = []
        now_colums = X.columns
        for feature in now_colums:
            ratio = 1 - (X[feature].count() / X[feature].shape[0])
            if ratio >= threshold:
                self.features_above_threshold_nan.append(feature)

        y = X.drop(self.features_above_threshold_nan, axis=1, inplace=inplace)

        if verbose and threshold > 1.0:
            raise_PasoError(" threshold gt 1.0 , threshold: {}".format(threshold))

        if verbose:
            logger.info(
                "Delete_NA_Features: {}".format(self.features_above_threshold_nan)
            )

        self.transformed_Delete_NA_Features = True

        if inplace:
            return X
        else:
            return y

    ########## 3
    def calculate_NaN_ratio(
        self, X: pd.DataFrame, inplace: bool = True, verbose: bool = True
    ) -> pd.DataFrame:
        """
        For a row with a large ratio of missing values (an observation)
        renders it statistically irrelevant. However, the row is not removed.
        instead the nulls/total_feature_count is calculated for each row and
        a new feature  ``NA_ratio``is added to the returned **pandas** dataframe.

        Note:
        Detecting and correcting for missing and outlier (good or bad)
        values is an evolving area of research.

        Parameters:
            X: DataFrame

            inplace : bool, default True
            If True, do operation inplace and return None.

            verbose: bool

        Returns:
                dataframe
                self.rows_mvr: features removed

        """
        total_row_count = X.shape[0]
        total_feature_count = X.shape[1]

        if verbose:
            logger.info("Calculate_NA_ratio")

        self.transformed_Calculate_NA_ratio = True

        self.column_missing_value_ratio = 1 - X.count(axis=0) / total_row_count

        if inplace:
            X["NaN_ratio"] = 1 - X.count(axis=1) / total_feature_count
            return X
        else:
            y = copy.deepcopy(X)
            y["NaN_ratio"] = X.isnull().sum(axis=1) / total_feature_count
            return y

    ########## 4
    def delete_Duplicate_Features(
        self,
        X: pd.DataFrame,
        ignore: List[str] = [],
        inplace: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        If a feature has the same values by index as another feature
        then one of those features should be deleted. The duplicate
        feature is redundant and will have no predictive power. 

        Duplicate features are quite common as an enterprise's
        database or data lake ages and different data sources are added.

        Parameters:
            X: (DataFrame)

            ignore: (list) default []
                The features (column names) not to eliminate.
                Usually this keywod argument is used for the target feature
                that is maybe present in the training dataset.

            ignore_NaN: bool

            verbose: bool

            inplace : bool, default True
            If True, do operation inplace and return None.

        Returns:
            X: X: (DataFrame)
                transformed X DataFrame

        Raises:

        Note: All NaN imputed or removed.

        """
        _Check_No_NA_Values(X)

        self.ignore = ignore

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
            if verbose:
                logger.info("Duplicate_Features_Removed: {}".format(str(drop_list)))

        self.transformed_delete_Duplicate_Features = True

        if inplace:
            return X
        else:
            return y

    ########## 5
    def delete_Features_with_Single_Unique_Value(
        self,
        X: pd.DataFrame,
        ignore: List[str] = [],
        inplace: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:

        """
        This method finds all the features which have only one unique value.
        The variation between values is zero. All these features are removed from
        the dataframe as they have no predictive ability.

        Parameters:
            Xarg: (DataFrame)
            ignore: (list) default []
                The features (column names) not to eliminate.
                Usually this keywod argument is used for the target feature
                that is maybe present in the training dataset.

            inplace : bool, default True
                If True, do operation inplace and return None.

            verbose: bool

        Returns:
            X: X: (DataFrame)
                transformed X DataFrame

        Raises:

        Note: All NaN imputed or removed.
        """

        _Check_No_NA_Values(X)

        efs = []
        n = 0
        for f in X.columns:
            if f in ignore:
                pass
            # weirdness where df has
            # 2 or more features with same name
            #               len(df[f].squeeze().shape) == 1) and
            elif X[f].nunique() == 1:
                efs.append(f)
                n += 1
        if n > 0:
            if verbose:
                logger.info(
                    "Eliminate_Single_Unique_Value_Features {}".format(str(efs))
                )
            for f in efs:
                y = X.drop(f, inplace=inplace, axis=1)

        self.transformed_Features_with_Single_Unique_Value = True

        if inplace:
            return X
        else:
            return y

    # 5nb.
    def delete_Features_with_All_Unique_Values(
        self,
        X: pd.DataFrame,
        ignore: List[str] = [],
        inplace: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:

        """
        This method finds all the features which have only number of unique values equal value count.
        The variation between values is zero. All these features are removed from
        the dataframe as they have no predictive ability.

        Parameters:
            Xarg: (DataFrame)
            ignore: (list) default []
                The features (column names) not to eliminate.
                Usually this keywod argument is used for the target feature
                that is maybe present in the training dataset.

            inplace : bool, default True
                If True, do operation inplace and return None.

            verbose: bool

        Returns:
            X: X: (DataFrame)
                transformed X DataFrame

        Raises:

        Note: All NaN imputed or removed.
        """

        _Check_No_NA_Values(X)

        efs = []
        n = 0
        for f in X.columns:
            if f in ignore:
                pass
            # weirdness where df has
            # 2 or more features with same name
            #               len(df[f].squeeze().shape) == 1) and
            elif X[f].nunique() == X[f].count():
                efs.append(f)
                n += 1
        if n > 0:
            if verbose:
                logger.info(
                    "delete_Features_with_All_Unique_Values {}".format(str(efs))
                )
            for f in efs:
                y = X.drop(f, inplace=inplace, axis=1)

        self.transformed_Features_with_Single_Unique_Value = True

        if inplace:
            return X
        else:
            return y

    ########## 6.a
    @staticmethod
    def statistics() -> List[str]:
        return [k for k in Cleaners._statistics]

    ############ 6.b
    def feature_Statistics(
        self,
        X: pd.DataFrame,
        statistics: str = "all",
        concat: bool = True,
        inplace: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        # todo row statistics
        """
        Calculate the statistics of each feature and returns a dataframe
        where each row is that statistic.
        This method can be used as an diagnostic tool (concat = False)
        that is used to decide if the sd or any other statistic  is too small
        and thus will have low predictive power.  It can also add onto the dataset
        (concat=True), with the target feature removed, each statistic helps descibe.
        the distribution ot the feature.  This will only make sense for those features
        whose values are numeric.

        Parameters:
            X: (DataFrame)

            statistics: (list,str) 'all' default
                Must be 'all'or a list of symbols in  Cleaners._statistics[

            concat: bool
                if concat is True, a new dataFrame is returned

            inplace : bool,

            verbose: bool


        Returns:
                dataframe

        Raises:

        Note: The end goal of any feature elimination is to increase speed and perhaps
        decrease the loss. are should be used before eliminating
        any feature and a 2nd opinion of the **SHAP** value should be used in order
        to reach a decision to remove a feature.

        Note: All NaN imputed or removed.


        """
        # work around automated review complaint
        if inplace:
            y = X
        else:
            y = X.copy()

        _Check_No_NA_Values(X)

        _must_be_list_tuple_int(statistics)
        if "all" in statistics:
            statistics = Cleaners._statistics[:-1]
        tmp_statistics = []
        for stat in statistics:
            if stat in Cleaners._statistics[:-1]:
                tmp_statistics.append(stat)
            else:
                raise_PasoError(
                    "\n One of {}\n is unknown statistic from the list of accepted statistics\n{}".format(
                        statistics, Cleaners._statistics[:-1]
                    )
                )
        statistics = tmp_statistics
        # by column stats
        self.column_stats = pd.DataFrame()
        for stat in statistics:
            m = eval("X." + stat + "(axis=0)")
            self.column_stats = pd.concat([self.column_stats, m], axis=1)
        self.column_stats.columns = statistics

        # by row stats
        s = pd.DataFrame()
        for stat in statistics:
            m = pd.DataFrame(eval("X." + stat + "(axis=1)"), columns=[stat])
            s = pd.concat([s, m], axis=1)

        if verbose:
            logger.info("Features_Statistics {}".format(statistics))

        self.transformed_Features_Statistics = True

        if concat:
            y = pd.concat([y, s], axis=1)
            return y
        else:
            return s

    # 7
    def boolean_to_integer(
        self, X: pd.DataFrame, inplace: bool = True, verbose: bool = True
    ) -> pd.DataFrame:
        """
        transform spurious bool  features and values from dataset. Encoding and scaling
        and other data-set preprocessing should not be done here.

        parameters:
            X: (DataFrame) are independent features of dataset

        keywords:
            inplace : bool,
                ignore,

            verbose: bool

        returns:
                X: (DataFrame)

        Note:
           Called once, clean can be called more than once,
           except after most_frequent because np.NaNs will be gone.

        """
        if not inplace:
            inplace = True

        # change boolean from whatever to 0,1
        bools = [f for f in X.columns if X[f].nunique() == 2]
        for feature in bools:
            boolean_values = list(X[feature].unique())
            X[feature].replace(to_replace=boolean_values[0], value=0, inplace=inplace)
            X[feature].replace(to_replace=boolean_values[1], value=1, inplace=inplace)

        self.transformed_booleans = True

        if verbose:
            logger.info("\nboolean_to_integer features: {}".format(bools))

        return X

    ########## 8
    def feature_Feature_Correlation(
        self, X: pd.DataFrame, method: str = "pearson", verbose: bool = True
    ) -> pd.DataFrame:
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

        Parameters:
            X: (DataFrame)

            method : {‘pearson’, ‘kendall’, ‘spearman’} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarrays

            verbose: bool

        Returns:
                Correlation of X dataframe

        Raises:

        Note: All NaN imputed or removed.

        """
        self.corr_method = method
        if verbose:
            logger.info("Correlation method: {}", self.corr_method)
        corr = X.corr(method=self.corr_method)
        self.transformed_Feature_Feature_Correlation = True
        return corr

    def plot_corr(
        self,
        X: pd.DataFrame,
        kind: str = "numeric",
        mirror: bool = False,
        xsize: float = 10,
        ysize: float = 10,
    ) -> None:
        """"
        Plot of correlation matrix.

        Parameters:
            X: (DataFrame)

            kind : {‘numeric’, ‘visual’}

            mirror : bool, default False
                If True,show oppostie (mirror) half of matrix.

            xsize:  float

            ysize: float

        Note: All NaN imputed or removed.

        """

        # todo put in EDA module

        def plot_corr_numeric(corr):
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

        def heatmap(x, y, **kwargs):
            if "color" in kwargs:
                color = kwargs["color"]
            else:
                color = [1] * len(x)

            if "palette" in kwargs:
                palette = kwargs["palette"]
                n_colors = len(palette)
            else:
                n_colors = 256  # Use 256 colors for the diverging color palette
                palette = sns.color_palette("Blues", n_colors)

            if "color_range" in kwargs:
                color_min, color_max = kwargs["color_range"]
            else:
                color_min, color_max = (
                    min(color),
                    max(color),
                )  # Range of values that will be mapped to the palette, i.e. min and max possible correlation

            def value_to_color(val):
                if color_min == color_max:
                    return palette[-1]
                else:
                    val_position = float((val - color_min)) / (
                        color_max - color_min
                    )  # position of value in the input range, relative to the length of the input range
                    val_position = min(
                        max(val_position, 0), 1
                    )  # bound the position betwen 0 and 1
                    ind = int(
                        val_position * (n_colors - 1)
                    )  # target index in the color palette
                    return palette[ind]

            if "size" in kwargs:
                size = kwargs["size"]
            else:
                size = [1] * len(x)

            if "size_range" in kwargs:
                size_min, size_max = kwargs["size_range"][0], kwargs["size_range"][1]
            else:
                size_min, size_max = min(size), max(size)

            size_scale = kwargs.get("size_scale", 500)

            def value_to_size(val):
                if size_min == size_max:
                    return 1 * size_scale
                else:
                    val_position = (val - size_min) * 0.99 / (
                        size_max - size_min
                    ) + 0.01  # position of value in the input range, relative to the length of the input range
                    val_position = min(
                        max(val_position, 0), 1
                    )  # bound the position betwen 0 and 1
                    return val_position * size_scale

            if "x_order" in kwargs:
                x_names = [t for t in kwargs["x_order"]]
            else:
                x_names = [t for t in sorted(set([v for v in x]))]
            x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

            if "y_order" in kwargs:
                y_names = [t for t in kwargs["y_order"]]
            else:
                y_names = [t for t in sorted(set([v for v in y]))]
            y_to_num = {p[1]: p[0] for p in enumerate(y_names)}

            plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x10 grid
            ax = plt.subplot(
                plot_grid[:, :-1]
            )  # Use the left 14/15ths of the grid for the main plot

            marker = kwargs.get("marker", "s")

            kwargs_pass_on = {
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "color",
                    "palette",
                    "color_range",
                    "size",
                    "size_range",
                    "size_scale",
                    "marker",
                    "x_order",
                    "y_order",
                ]
            }

            ax.scatter(
                x=[x_to_num[v] for v in x],
                y=[y_to_num[v] for v in y],
                marker=marker,
                s=[value_to_size(v) for v in size],
                c=[value_to_color(v) for v in color],
                **kwargs_pass_on
            )
            ax.set_xticks([v for k, v in x_to_num.items()])
            ax.set_xticklabels(
                [k for k in x_to_num], rotation=45, horizontalalignment="right"
            )
            ax.set_yticks([v for k, v in y_to_num.items()])
            ax.set_yticklabels([k for k in y_to_num])

            ax.grid(False, "major")
            ax.grid(True, "minor")
            ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
            ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

            ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
            ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
            ax.set_facecolor("#F1F1F1")

            # Add color legend on the right side of the plot
            if color_min < color_max:
                ax = plt.subplot(
                    plot_grid[:, -1]
                )  # Use the rightmost column of the plot

                col_x = [0] * len(palette)  # Fixed x coordinate for the bars
                bar_y = np.linspace(
                    color_min, color_max, n_colors
                )  # y coordinates for each of the n_colors bars

                bar_height = bar_y[1] - bar_y[0]
                ax.barh(
                    y=bar_y,
                    width=[5] * len(palette),  # Make bars 5 units wide
                    left=col_x,  # Make bars start at 0
                    height=bar_height,
                    color=palette,
                    linewidth=0,
                )
                ax.set_xlim(
                    1, 2
                )  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
                ax.grid(False)  # Hide grid
                ax.set_facecolor("white")  # Make background white
                ax.set_xticks([])  # Remove horizontal ticks
                ax.set_yticks(
                    np.linspace(min(bar_y), max(bar_y), 3)
                )  # Show vertical ticks for min, middle and max
                ax.yaxis.tick_right()  # Show vertical ticks on the right

        def plot_corr_visual(data, size_scale=500, marker="s"):
            corr = pd.melt(data.reset_index(), id_vars="index")
            corr.columns = ["x", "y", "value"]
            heatmap(
                corr["x"],
                corr["y"],
                color=corr["value"],
                color_range=[-1, 1],
                palette=sns.diverging_palette(20, 220, n=256),
                size=corr["value"].abs(),
                size_range=[0, 1],
                marker=marker,
                x_order=data.columns,
                y_order=data.columns[::-1],
                size_scale=size_scale,
            )

        _Check_No_NA_Values(X)
        corr = X.corr()
        _, _ = plt.subplots(figsize=(xsize, ysize))
        _ = sns.diverging_palette(220, 10, as_cmap=True)
        if kind == "numeric":
            plot_corr_numeric(corr)
        elif kind == "visual":
            plot_corr_visual(corr)
        else:
            raise_PasoError("plot_corr, unknow kind:{}".format(kind))

        plt.show()

    ########## 8
    def delete_Features(
        self,
        X: pd.DataFrame,
        features: List[str] = [],
        verbose: bool = True,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """

        This class finds all the features which have only one unique value.
        The variation between values is zero. All these features are removed from
        the dataframe as they have no predictive ability.

        Parameters:
            X: (DataFrame)

            features: (list) default []
                The features (column names) to eliminate.

            inplace : bool,

            verbose: bool

        Returns:
            X: X: (DataFrame)
                transformed X DataFrame
        Raises:

        Note:

        """
        self.delete_features = features
        y = X.drop(self.delete_features, axis=1, inplace=inplace)

        if verbose:
            logger.info("delete_Features {}".format(self.delete_features))

        self.transformed = True

        if inplace:
            return X
        else:
            return y

    ########## 9
    def delete_Features_not_in_train_or_test(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        ignore: List[str] = [],
        verbose: bool = True,
        inplace: bool = True,
    ) -> pd.DataFrame:

        """
        If the train or test datasets have features the other
        does not, then those features will have no predictive
        power and should be removed from both datasets.
        The exception being the target feature that is present
        in the training dataset of a supervised problem. 
        Duplicate features are quite common as an enterprise's
        database or datalake ages and different data sources are added.

        Parameters:
            train: (DataFrame)

            test:  (DataFrame)

            ignore: (list) default []
                The features (column names) not to eliminate.
                Usually this keywod argument is used for the target feature
                that is maybe present in the training dataset.

            inplace: bool

            verbose: bool

        Returns:
            X: X: (DataFrame)
                EliminateUnviableFeatures transformed train DataFrame
            Y: Y: (DataFrame)
                EliminateUnviableFeatures transformed test DataFrame

        Raises:

        Note: All NaN imputed or removed.

        """
        self.ignore = ignore
        if inplace:
            X = train
            y = test
        else:
            X = train.copy()
            y = test.copy()

        _Check_No_NA_Values(X)
        _Check_No_NA_Values(y)

        rem = set(X.columns).difference(set(y.columns))
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
                    "Clean_Features_in_X: {}".format(str(x_features_cleaned_efs))
                )

        rem = set(y.columns).difference(set(X.columns))
        y_features_cleaned_efs = []
        y_features_cleaned = 0
        if len(rem) >= 1:
            for f in rem:
                if f not in self.ignore:
                    y.drop(f, inplace=True, axis=1)
                    y_features_cleaned_efs.append(f)
                    y_features_cleaned += 1
        if y_features_cleaned > 0:
            if verbose:
                logger.info(
                    "Clean_Features_not_in_y {}".format(str(y_features_cleaned_efs))
                )

        if len(X.columns) == 0:
            logger.error(
                "Clean_Features_not_in_X_or_test:transform:X and Y are orthogonal."
            )
            raise PasoError()

        if inplace:
            return None
        else:
            return X, y


class Balancers(pasoFunction):
    """
    Currently, **paso** supports only the imbalanced-learn
    package. This package is very comprehensive with
    examples on how to transform (clean) different types of
    imbalanced class data.

    Note:

        Warning:
            Only **SMOTEC** can balance datasets with categorical features. All
            others will accept a dataset only with continuous features.

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

    __Balancers__ = {
        "RanOverSample": RandomOverSampler,
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

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs):
        """

        """

        super().__init__()

    @staticmethod
    def balancers():
        """
        Parameters:
            None

        Returns:
            List of available class Balancers ames.
        """
        return list(Balancers.__Balancers__.keys())

    @pasoDecorators.TTWrapXy(array=False)
    def transform(self, X, y, verbose=True, **kwargs):
        """
        Parameters:
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.

        Returns: ( Balanced)
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.
        """

        # create instance of this particular learner
        # checks for non-optional keyword
        if self.kind_name not in Balancers.__Balancers__:
            raise_PasoError(
                "transform; no Balancer named: {} not in Balancer.__Balancers__: {}".format(
                    self.kind_name, Balancers.__Balancers__.keys()
                )
            )
        else:
            self.model = Balancers.__Balancers__[self.kind_name](
                **self.kind_name_kwargs
            )

        # workaround in case all classes are equal

        uniqueValues, _, occurCount = np.unique(
            y, return_index=True, return_counts=True
        )

        if np.max(occurCount) == np.min(occurCount):
            return X, y

        # workaround in case all classes are equal

        uniqueValues, _, occurCount = np.unique(
            y, return_index=True, return_counts=True
        )

        self.n_class = len(uniqueValues)
        self.class_names = _array_to_string(uniqueValues)

        X_result, y_result = self.model.fit_resample(X.to_numpy(), y)

        X = pd.DataFrame(X_result, columns=X.columns)

        if verbose:
            logger.info("Balancer")
        self.balanced = True

        return X, y_result


class Augmenters(pasoFunction):
    """
    Currently, **paso** supports claas stutured data.

    Note:
    """

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs):
        """
            Parameters:
                name:
                description_filepath:
            Returns:
                self
        """
        super().__init__()

    @pasoDecorators.TTWrapXy(array=False)
    def transform(self, X, y, verbose=True, **kwargs):
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
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.
            ratio: (float) must be [0.0,1.0]. setting to 1.0 means dunle size if dataser
                    size = (1 + ratio)

        Returns: ( Augmented)
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.

            Raises:

            Note:
                Because of integer roundoff. the ratio increased may not be exact.
                Also, ratio of 0.0 or less indicates balance only no augmentation.

        """

        kwa = "ratio"
        self.ratio = _dict_value(kwargs, kwa, None)
        _check_non_optional_kw(
            self.ratio,
            "ratio keyword pair not specified in Balancer:.ratio {}".format(kwargs),
        )

        # create instance of this particular learner
        # checks for non-optional keyword
        if self.kind_name not in Balancers.__Balancers__:
            raise_PasoError(
                "transform; no Augmenter named: {} not in Balancer.__Balancers__: {}".format(
                    self.kind_name, Balancers.__Balancers__.keys()
                )
            )
        else:
            self.model = Balancers.__Balancers__[self.kind_name](
                **self.kind_name_kwargs
            )

        # balance before augmentation
        X, y = Balancers(description_filepath=self.description_filepath).transform(
            X, y, verbose=False
        )
        # 0.0or less indicates balance only no augmentation
        if self.ratio <= 0.0:
            logger.warning("ratio lt 0. just returning X,y balanced")
            return X, y
        elif self.ratio > 1.0:
            raise_PasoError("Ratio<= 1.0 was: {}".format(self.ratio))

        # 3
        # calcuate before balancer creates augment data (pusedo data)
        # ratio is the multiplier of max sized class
        # augment will first balance so that 1.0 will result
        # in a size
        target = "TaRgEtx403856789"
        Xy_to_DataFrame(X, y, target)

        self.class_names = _array_to_string(X[target].unique())
        each_class_count = X.groupby([target]).count()
        max_class_sizes_s = each_class_count.iloc[:, 0]
        max_class_size = max_class_sizes_s.max()
        stub_size = int(max_class_size)
        highest_class_arg = max_class_sizes_s.argmax()

        # only sample as ratio can not be bigger than 1.0
        stub = X[X[target] == highest_class_arg].sample(stub_size)
        # 4
        X = X.append(stub)
        X, y = DataFrame_to_Xy(X, target)
        X, y = Balancers(description_filepath=self.description_filepath).transform(
            X, y, verbose=False
        )

        # 5the stub is subtracted
        Xy_to_DataFrame(X, y, target)
        X.drop(X.index[stub.index], axis=0, inplace=True)
        # 6 and the dataset is rebalanced.
        X, y = DataFrame_to_Xy(X, target)
        X, y = Balancers(description_filepath=self.description_filepath).transform(
            X, y, verbose=False
        )
        if verbose:
            logger.info("Augmenter ratio: {}".format(self.ratio))
        self.augmented = True
        return X, y
