# !/usr/bin/env python
# -*- coding: utf-8 -*-
from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.binary import BinaryEncoder

# from category_encoders.count import CountEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.basen import BaseNEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.cat_boost import CatBoostEncoder


# paso imports
from paso.base import pasoModel, PasoError
from paso.base import toDataFrame, pasoDecorators
from loguru import logger

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

#######


class EmbeddingVectorEncoder(BaseEstimator, TransformerMixin):
    """

    Attributes:
        None
    """

    def __init__(self):
        self.coefs_ = []  # Store tau for each transformed variable

    #        self.verbose = verbose

    def _reset(self):
        self.coefs_ = []  # Store tau for each transformed variable

    def fit(self, X, **kwargs):
        return self.train(X, **kwargs)

    def train(self, X, inplace=False, **kwargs):
        """
        Args:
            X (dataframe):
                Calculates BoxCox coefficients for each column in X.
            
        Returns:
            self (model instance)
        
        Raises:
            ValueError will result of not 1-D or 2-D numpy array, list or Pandas Dataframe.
            
            ValueError will result if has any negative value elements.
            
            TypeError will result if not float or integer type.

        """

        for x_i in X.T:
            self.coefs_.append(boxcox(x_i)[1])
        return self

    def transform(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def predict(self, X, inplace=False, **kwargs):
        # todo: transform dataframe to numpy and back?
        """
        Transform data using a previous ``.fit`` that calulates BoxCox coefficents for data.
                
        Args:
            X (np.array):
                Transform  numpy 1-D or 2-D array distribution by BoxCoxScaler.

        Returns:
            X (np.array):
                tranformed by BoxCox

        """

        return np.array(
            [boxcox(x_i, lmbda=lmbda_i) for x_i, lmbda_i in zip(X.T, self.coefs_)]
        ).T

    # is there a way to specify with step or does does step need enhancing?
    def inverse_transform(self, y):
        """
        Recover original data from BoxCox transformation.
        """

        return np.array(
            [
                (1.0 + lmbda_i * y_i) ** (1.0 / lmbda_i)
                for y_i, lmbda_i in zip(y.T, self.coefs_)
            ]
        ).T

    def inverse_predict(self, y, inplace=False):
        """
        Args:
            y: dataframe

        Returns:
            Dataframe: Recover original data from BoxCox transformation.
        """
        return self.inverse_transform(y)


########
class EncoderVariables(object):
    """
    Holds class variables
    """

    _category_encoders__version__ = "2.0.0"

    Encoder_Dict = {}

    category_encoders__all__ = [
        "BackwardDifferenceEncoder",
        "BinaryEncoder",
        "HashingEncoder",
        "HelmertEncoder",
        "OneHotEncoder",
        "OrdinalEncoder",
        "SumEncoder",
        "PolynomialEncoder",
        "BaseNEncoder",
        "LeaveOneOutEncoder",
        "TargetEncoder",
        "WOEEncoder",
        "MEstimateEncoder",
        "JamesSteinEncoder",
        "CatBoostEncoder",
    ]
class Encoder(pasoModel):
    """
    Parameters:
        encoderKey: (str) One of Encoder.encoders()

        verbose: (str) (default) True, logging off (``verbose=False``)

    Note:
        **Encode**
    """

    def __init__(self, encoderKey=None, verbose=False, *args, **kwargs):
        super().__init__()
        if encoderKey in EncoderVariables.Encoder_Dict:
            Encoder = EncoderVariables.Encoder_Dict[encoderKey](*args)
        else:
            logger.error("paso:encoder: No Encoder named: {} found.".format(encoderKey))
            raise PasoError()

        self.encoderKey = encoderKey
        self.model = Encoder
        validate_bool_kwarg(verbose, "verbose")
        self.verbose = verbose

    def encoders(self):
        """
        Parameters:
            None

        Returns:
            List of available encoders names.
        """
        return list(EncoderVariables.Encoder_Dict.keys())

    @pasoDecorators.TrainWrap(array=True)
    def train(self, X, inplace=False, **kwargs):
        """

        Parameters:
            Xarg:  pandas dataFrame #Todo:Dask numpy

            inplace: (CURRENTLY IGNORED)
                    False (boolean), replace 1st argument with resulting dataframe
                    True:  (boolean) ALWAYS False
            
        Returns:
            self
        """

        self.model.fit(X, **kwargs)

        return self

    @pasoDecorators.PredictWrap(array=False)
    def predict(self, X, inplace=False, **kwargs):
        """
        Parameters:
            X:  pandas dataFrame #Todo:Dask numpy

            inplace: (CURRENTLY IGNORED)
                    False (boolean), replace 1st argument with resulting dataframe
                    True:  (boolean) ALWAYS False
            
        Returns:
            (DataFrame): transform X
        """

        self.f_x = self.model.transform(X, **kwargs)
        return self.f_x

    def inverse_predict(self, Xarg, inplace=False, **kwargs):
        """
        Args:
            Xarg (array-like): Predictions of different models for the labels.

        Returns:
            (DataFrame): inverse of Xarg
        """
        X = Xarg.values
        if self.trained and self.predicted:
            X = self.model.inverse_transform(X)
            if self.verbose:
                logger.info("Scaler:inverse_transform:{}".format(self.encoderKey))
            return toDataFrame().transform(X, labels=Xarg.columns, inplace=False)
        else:
            raise pasoError(
                "scale:inverse_transform: must call train and predict before inverse"
            )

    def load(self, filepath=None):
        """
       Can not load an encoder model.

        Parameters:
            filepath: (str)
                ignored

        Raises:
                raise PasoError(" Can not load an encoder model.")
        """

        logger.error("Can not load an encoder model.")
        raise PasoError()

    def save(self, filepath=None):
        """
        Can not save an encoder model.

        Parameters:
            filepath: (str)
                ignored

        Raises:
                raise PasoError(" Can not save an encoder model.")

        """

        logger.error("Can not save an encoder model.")
        raise PasoError()


########
# initilize wth all encoders
for encoder in EncoderVariables.category_encoders__all__:
    EncoderVariables.Encoder_Dict[encoder] = eval(encoder)
# add new encoders
EncoderVariables.Encoder_Dict["EmbeddingVectorEncoder"] = EmbeddingVectorEncoder
