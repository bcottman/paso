import pandas as pd
from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm

# paso imports
from paso.base import pasoFunction, pasoDecorators
from loguru import logger
import sys

# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#
#
class toCategory(pasoFunction):
    """
        Transforms any boolean, object or integer numpy arry, list, tuple or
        any pandas dataframe or series feature(s) type(s) to category type(s).
        The exception is continuous (``float`` or ``datetime``) which are
        returned as is. If you want to convert continuous or datimw types to
        category then use ``ContinuoustoCategory`` or ``DateTimetoCategory``
        before **paso** (step) ``toCategory``.
        
        Note:
            Assumes paso ``EliminateUnviableFeatures()`` and other data
            cleaning steps (such as removal of Null and NA values) has
            been done previous to this step.

            ``datetime`` features should call ``toDatetimeComponents()`` 
            previous to this step so that ``datetime`` components (which are of type 
            ``np.nmnber``) can be converted to ``category``. The default 
            behavior of this step is NOT to convert ``datetime`` to ``category``.
        
        
        Parameters:
            verbose: (boolean) default: True
                Logging on/off (``verbose=False``)

    """

    def __init__(self, verbose=True):

        super().__init__()

        validate_bool_kwarg(verbose, "verbose")
        self.verbose = verbose

    @pasoDecorators.TransformWrap()
    def transform(
        self, Xarg, inplace=False, boolean=True, integer=True, objects=True, **kwargs
    ):
        """
        Transforms any boolean, object or integer numpy arry, list, tuple or
        any pandas dataframe or series feature(s) type(s) to category type(s).

        Note:
            No matter the type of ``X`` it
            will be returned as ``DataFrame``.

            If ``X`` is of type dataframe, ``inplace=False``
        
        Parameters:
            X: (dataFrame,numpy array,list)

            inplace: (boolean) default: False,
                replace 1st argument with resulting dataframe.


            boolean: (boolean) default:True
                If ``True`` will convert to ``category`` type.

            integer: (boolean) default: True
                If ``True`` will convert to ``category`` type.

            objects: (boolean) default: True
                If ``True`` will convert to ``category`` type.

        Returns:
            (DataFrame)
        
        Raises:
            TypeError('toCategory:_transform:ufit must be invoked before transform.')
       """

        validate_bool_kwarg(boolean, "boolean")
        self.boolean = boolean
        validate_bool_kwarg(integer, "integer")
        self.integer = integer
        validate_bool_kwarg(objects, "objects")
        self.object = objects

        for feature in Xarg.columns:
            if Xarg[feature].dtype == np.bool and self.boolean:
                Xarg[feature] = Xarg[feature].astype("category")
                if self.verbose:
                    logger.info(
                        "toCategory boolean feature converted : {}".format(feature)
                    )
            elif Xarg[feature].dtype == np.object and self.object:
                Xarg[feature] = Xarg[feature].astype("category")
                if self.verbose:
                    logger.info(
                        "toCategory object(str) feature converted : {}".format(feature)
                    )
            elif Xarg[feature].dtype == np.integer and self.integer:
                Xarg[feature] = Xarg[feature].astype("category")
                if self.verbose:
                    logger.info(
                        "toCategory integer feature converted : {}".format(feature)
                    )
            else:
                pass

        return Xarg
