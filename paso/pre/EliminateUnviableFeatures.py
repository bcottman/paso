# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas.util._validators import validate_bool_kwarg
import warnings
warnings.filterwarnings("ignore")

# paso import
from paso.base import _Check_No_NA_Values
from paso.base import pasoFunction, pasoDecorators
from loguru import logger

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#
class EliminateUnviableFeatures(pasoFunction):
    """
    1. check if any feature has NA values
        - if there are any NA, then raise error.
        - since values for NA are so domain dependent,
            it is required that all be removed before calling this paso transformer.

    1. Elimnate Features not found in train and test

    1. Eliminate duplicate features ; check all values
        - Eliminate single unique value features
        - Eliminate low variance features:  > std/(max/min)
        - All deleted features are logged

    """
    def __init__(self, verbose=True):
        """
            Parameters:

                verbose: (str) (default) True, logging on/off (``verbose=False``)
        """

        super().__init__()
        validate_bool_kwarg(verbose,'verbose')
        self.verbose = verbose
    # 1
    def _Eliminate_Features_not_found_in_train_and_test(self, train, test):
        """
        Note:
            ``EliminateUnviableFeatures.transform(X,X)`` where ``X'' signifies identical
            arguments, then ``X`` will be returned as is, as expected.

             Internal function. Should never be called directly.
        """
        rem = set(train.columns).difference(set(test.columns))
        efs = []
        n = 0
        if len(rem) >= 1:
            for f in rem:
                train.drop(f, inplace=True, axis=1)
                efs.append(f)
                n += 1
        if n > 0:
            if self.verbose: logger.info("_Eliminate_Features_not_found_in_TEST: {}".format(str(efs)))

        rem = set(test.columns).difference(set(train.columns))
        efs = []
        n = 0
        if len(rem) >= 1:
            for f in rem:
                test.drop(f, inplace=True, axis=1)
                efs.append(f)
                n += 1
        if n > 0:
            if self.verbose: logger.info("_Eliminate_Features_not_found_in_TRAIN {}".format(str(efs)))

        return True

    # 2
    def _Eliminate_Duplicate_Features(self, df):
        k = [f for f in df.columns]
        equal = dict(zip(k, k))
#        if df.columns[0] in equal:
#            del equal[df.columns[0]]
        #            import pdb; pdb.set_trace() # debugging starts here
        for nth, f in enumerate(df.columns[0:-1]):
            if f in self.ignore: break
            for mth, g in enumerate(df.columns[nth + 1 :]):
                if (df[f].values != df[g].values).any():
                    if f in equal:
                        del equal[f]  # stop looping as unequal
 #                       break
        if df.columns[-1] in equal:
            del equal[df.columns[-1]]  # stop looping as unequal

        #            import pdb; pdb.set_trace() # debugging starts here
        drop_list = list(equal.keys())
        try:  # for some whacko reason there are more than 2 identical columns (sometimes)
            df.drop(columns=drop_list, inplace=True, index=1)
            if self.verbose: logger.info("_Eliminate_Duplicate_Features {}".format(str(drop_list)))
        except:
            pass

        return True

    # 3
    def _Eliminate_Single_Unique_Value_Features(self, df):
        efs = []
        n = 0
        #        import pdb; pdb.set_trace() # debugging starts here
        for f in df.columns:
            if f in self.ignore: pass
            # weirdness where df has
            # 2 or more features with same name
            #               len(df[f].squeeze().shape) == 1) and
            elif df[f].nunique() == 1:
                efs.append(f)
                n += 1
        if n > 0:
            if self.verbose: logger.info(
                "Eliminated Eliminate_Single_Unique_Value_Features {}".format(str(efs)))
            for f in efs:
                df.drop(f, inplace=True, axis=1)

        return True

    # 4
    def _Eliminate_Low_Variance_Features(self, df):
        """
        Note:
            There must be At least 10 or more values in a feature, or variance is not
            calculated, and passed arguments(by this point tranformed into dataframe(s)
            are returned as is.
        """

        #        import pdb; pdb.set_trace() # debugging starts here
        efs = []
        n = 0
        for f in df.columns:
            if f in self.ignore:
                pass
            elif (type(df[f].iloc[0]) == np.float_) or (type(df[f].iloc[0]) == np.int_):
                v = df[f].std() / df[f].max()
                #                import pdb; pdb.set_trace() # debugging starts here
                if df[f].shape[0] <= 10:  # must have more than 10 values in feature
                    if self.verbose: logger.info(
                        "NOT** _Eliminate_Low_Variance_Features Before len lt 10 {}".format(
                            f
                        )
                    )
                elif (
                    (f not in self.ignore)
                    and (len(df[f].squeeze().shape) == 1)
                    and (v < self.SD_LIMIT)
                ):
                    efs.append([f, v])
                    n += 1

        if n > 0:
            if self.verbose: logger.info(
                "Eliminated _Eliminate_Low_Variance_Features {}".format(str(efs))
            )
            for f in efs:
                df.drop(f[0], inplace=True, axis=1)

        return True

    @pasoDecorators.TransformWrap
    def transform(self, Xarg, Yarg=None, inplace=False, ignore=[], SD_LIMIT=0.00001):
        """
        Parameters:
            Xarg: (DataFrame)
            Yarg: (DataFrame): default None
            ignore: (list) default []
                The features(column names) not to eliminate unless **Duplicate**
            SD_LIMIT  (float) default: .000001
        Returns:
            X: X: (DataFrame)
                EliminateUnviableFeatures transformed train DataFrame
            Y: Y: (DataFrame)
                EliminateUnviableFeatures transformed test DataFrame
                
       Raises:
            ValueError: if any feature has NA values.
            ValueError: if X and Y arguments have orthogonal features.
            ValueError: if X and Y arguments after _Eliminate_Single_Unique_Value_Features has no features.  
            ValueError: if X and X_test arguments after _Eliminate_Low_Variance_Features has no features. 
        """
        self.ignore = ignore
        self.SD_LIMIT = SD_LIMIT
        if self.inplace:
            Y = Yarg
        else:
            if Yarg is None:
                Y = None
            else:
                Y = Yarg.copy()

        if not Y is None:
            _Check_No_NA_Values(Y)

        for nth, df in enumerate([Xarg, Y]):
            if df is None: break
            # 0
#            import pdb; pdb.set_trace()  # debugging starts here
            if _Check_No_NA_Values(df):
                # 4
                result = self._Eliminate_Low_Variance_Features(df)
                if not result:
                    raise TypeError(
                        "After _Eliminate_Low_Variance_Features:transform:df: has no features."
                    )
                # 2
                result = self._Eliminate_Duplicate_Features(df)
                if not result:
                    raise TypeError(
                        "After _Eliminate_Duplicate_Features:transform:df: has no features."
                    )
                # 3 ,unless SDLIMIT < 0, _Eliminate_Duplicate_Features will eminate single value
                result = self._Eliminate_Single_Unique_Value_Features(df)
                if not result:
                    raise TypeError(
                        "After _Eliminate_Single_Unique_Value_Features:transform:df: has no features."
                    )
            else:
                raise ValueError(
                    "EliminateUnviableFeatures:transform:X_test: \
                        X DataFrame has NaN values: "
                    )
        if Y is None:
            pass
        else:
            # 1
            result = self._Eliminate_Features_not_found_in_train_and_test(Xarg, Y)
            if not result:
                raise TypeError(
                    "_Elimnate_Features_not_found_in_train_and_test:transform:X and Y are orthogonal."
                )

        if self.cache: self.f_x = Xarg,Y
        self.transformed = True

        return Xarg,Y