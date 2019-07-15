import pandas as pd
from tqdm import tqdm
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")

# paso imports
from paso.base import pasoModel, PasoError,NameToClass
from paso.base import Param, pasoDecorators
from loguru import logger
import sys

# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#

class Learner(pasoModel):
    """
    Currently, **paso** supports only the sklearn RandomForest
    package. This package is very comprehensive with
    examples on how to train ad predict different types of
    modelss data.

    Note:

    Warning:

    """
    @pasoDecorators.InitWrap(narg=4)
    def __init__(self, name, category,ontological_file_path, **kwargs):

        """
        Parameters:
            modelKey: (str) On
            verbose: (str) (default) True, logging off (``verbose=False``)

        Note:

        """
        super().__init__()
        self.name = name
        self.category = category
        self.ontological_file_path = ontological_file_path

    def learners(self):
        """
        Parameters:
            None

        Returns:
            List of available models names.
        """
        return list(NameToClass.__learners__.keys())

    @pasoDecoratorsnarg.TrainWrap(array=True)
    def train(self, X, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            X:  pandas dataFrame 


        Returns:
            self
        """

        def divide_dict(d1, n):
            for k in d1.keys():
                d1[k] = d1[k]/n
            return d1


        def add_dicts(d1, d2):
            for k in d1.keys():
                if k in d2.keys():
                    d1[k] = d1[k] + d2[k]
            return {**d2, **d1}

#       self.metric is a list of length cv, where each element is a dict of metrics
        self.metric_valid = []
        for cv in range(self.modelDict['cv']):
            #todo support crosss-valid
            if self.modelDict['verbose']:
                logger.info('cross-validation [{}] of [{}]'.format(cv,self.modelDict['cv']))
            metric_average  = {}
            average_rounds= self.modelDict['average_rounds']
            if self.modelDict['verbose']:
                logger.info('The number averaging rounds: {}'.format(average_rounds))
            for average_round in tqdm(range(average_rounds)):
                if self.modelDict['valid']:


                    train_test_split_args = (self.X_train,self.y_train)
                    X_train, X_valid, y_train, y_valid = train_test_split(*train_test_split_args
                                                                            ,**self.train_test_split_kwargs)
#                    X_train, X_valid, y_train, y_valid = train_test_split(self.X_train,self.y_train,test_size=0.2)
                    self.model.fit(X_train, y_train)
                    y_pred = self.model.predict(X_valid)

                    if self.modelDict['metric' ] == 'class':
                        metric_average = add_dicts(metric_average,self.__metric_class(y_pred,y_valid))
                    elif self.modelDict['metric' ] == 'probability':
                        metric_average =  add_dicts(metric_average,self.__metric_probability(y_pred, y_valid))

            if self.modelDict['valid']:
                self.metric_valid.append(divide_dict(metric_average,average_rounds))

        if self.modelDict['metric']:
            if self.modelDict['verbose']:
                logger.info('metrics of error evaluated for model: {}'.format(self.modelKey))
        else:
            logger.info('metrics of error not known: {} or model: {}'.format(self.modelDict['metric'], self.modelKey))
            raise PasoError('metrics of error not known: {} or model: {}'.format(self.modelDict['metric'], self.modelKey))
            self.X_valid = None
            self.y_valid = None

        return self

    def __metric_class(self,y_pred, y_act):
        precision = metrics.precision_score(y_act, y_pred, average="macro")
        recall = metrics.recall_score(y_act, y_pred, average="macro")
        f1 = metrics.f1_score(y_act, y_pred, average="macro")
        cm = metrics.confusion_matrix(y_act, y_pred)

        return{'precision': precision ,'recall': recall ,'f1': f1, 'confusion_matrix': cm}

    def __metric_probability(self, y_pred, y_act):
        logLoss = metrics.log_loss(y_act, y_pred, eps=1e-15, normalize=False)

        return {'logLoss': logLoss}

    @pasoDecorators.PredictWrap(array=True)
    def predict(self, X_test, **kwargs):
        """
        Parameters:
            Xarg:  pandas dataFrame #Todo:Dask numpy

            inplace: (CURRENTLY IGNORED)
                    False (boolean), replace 1st argument with resulting dataframe
                    True:  (boolean) ALWAYS False
            
        Returns:
            (DataFrame): predict of X
        """

        if self.predict == 'predict':
            y_test_pred  = self.model.predict(X_test, **kwargs)
        elif  self.predict == 'predicta':
            y_test_pred = self.model.predicta(X, **kwargs)
        else:
            logger.error('Unknown model.predict: {}'.format(self.predict))
            raise PasoError('Unknown model.predict: {}'.format(self.predict))

        return y_test_pred

########
