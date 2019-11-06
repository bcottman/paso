# !/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


# sklearn.metrics imports
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, log_loss, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

# paso imports
from paso.base import pasoModel, NameToClass, _stat_arrays_in_dict
from paso.base import raise_PasoError, _exists_as_dict_value, _divide_dict
from paso.base import _array_to_string, pasoDecorators
from paso.base import _add_dicts, _merge_dicts


# from paso.base import _dict_value2
from loguru import logger

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
# class Learner
class Learners(pasoModel):
    """
    Currently, **paso** supports only the sklearn RandomForest
    package. This package is very comprehensive with
    examples on how to train ad predict different types of
    modelss data.

    Methods:
        train(X,y)
        evaluate(X,y) -> metrics->{}
            predict(y)
            predict_proba(y)
        plot_confusion_matrix(cm)
        cross_vadididation(X,y)
            train
            evaluate
        hyper_paranmeter_tune(X,y)
            cross_vadididation(X,y)
                train
                evaluate


    Note:

    Warning:

    """

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs):

        """
        Parameters:
            modelKey: (str) On
            verbose: (str) (default) True, logging off (``verbose=False``)

        Note:

        """
        super().__init__()
        self.debug = True
        self.model = None
        self.dataset_name = None
        self.model_name = None
        self.model_type = None
        #        self.n_early_stopping = 10
        #        self.metric_optimize = "f1_score"
        self.inputed = False
        self.cleaned = False
        self.cleaned_booleans = False
        self.encoded = False
        self.scaled = False
        self.trained = False
        self.predicted = False
        self.predicted_proba = False
        self.evaluated = False
        self.metrics = None
        self.metrics_scoring = None
        # {
        #     k: make_scorer(v[0], **v[1])
        #     for k, v in NameToClass.__metrics__["Classification"].items()
        # }
        self.metrics_names = None
        # [
        #     k for k in NameToClass.__metrics__["Classification"].keys()
        # ]
        self.cross_validated = False
        self.tuned_parameters = False


    @staticmethod
    def learners(self):
        """
        Parameters:
            None

        Returns:
            List of available models names.
        """
        return [k for k in NameToClass.__learners__.keys()]

    @pasoDecorators.TTWrapXy(array=False)
    def train(self, X, y, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            X:  pandas dataFrame, dataset[-taget]
            y: numpy array, target
        Returns:
            self
        """
        # currently support only one learner, very brittle parser
        if self.kind == {}:
            raise_PasoError(
                "keyword kind must be present at top level:{}:".format(
                    self.ontology_kwargs
                )
            )

        if self.kind_name not in NameToClass.__learners__:
            raise_PasoError(
                "Train; no operation named: {} in learners;: {}".format(
                    self.kind_name, NameToClass.__learners__.keys()
                )
            )
        else:
            self.model_name = self.kind_name
            self.model = NameToClass.__learners__[self.kind_name](
                **self.kind_name_kwargs
            )
            self.model.fit(X, y)
            self.model_type = self.type

        self.trained = True

        return self

    def predict(self, X):
        """
        parameters:
            X: (DataFrame\) column(s) are independent features of dataset

        Returns:
            y_pred: (numpy vector of ints)  predicted target.

        Warning:
            Assumes train has been called.


        """
        # enforce order of method calls
        if not self.trained:
            raise_PasoError("Must be 'fit' before predict. ")
        self.predicted = True
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        parameters:
            X: (DataFrame\) column(s) are independent features of dataset

        Returns:
            y_pred: (numpy vector of ints)  predicted target probability

        Warning:
            Assumes train has been called.
        """
        # enforce order of method calls
        if not self.trained:
            raise_PasoError("Must be ['fit'] before predict_proba. ")

        self.predicted_proba = True
        return self.model.predict_proba(X)

    def _parse_metrics(self, y, **kwargs):
        """"
        pasre the metics dict

        Parameters:

            y: (numpy vector )  target or dependent feature of dataset.

        """
        self.n_class = len(np.unique(y))
        self.class_names = _array_to_string(np.unique(y))

        self.metrics = {}
        if self.model_type in NameToClass.__metrics__:
            self.metrics_names = [
                k for k in NameToClass.__metrics__[self.model_type].keys()
            ]
            self.metrics_list = [
                k for k in NameToClass.__metrics__[self.model_type].values()
            ]
            self.metrics_f = [
                self.metrics_list[k][0] for k in range(len(self.metrics_names))
            ]
            self.metrics_f_type = [
                self.metrics_list[k][1] for k in range(len(self.metrics_names))
            ]
            self.metrics_f_kwargs = [
                self.metrics_list[k][2] for k in range(len(self.metrics_names))
            ]
            self.metrics_needs_proba = [
                self.metrics_list[k][3] for k in range(len(self.metrics_names))
            ]
            # this list will be shorter if non-binary.
            # ok but misleading as only used by cv
            self.metrics_scoring = [
                self.metrics_list[k][4]
                for k in range(len(self.metrics_names))
                if self.metrics_list[k][4]
                and (
                    (self.metrics_f_type[k] == NameToClass.BINARY)
                    and (self.n_class == NameToClass.NBINARY)
                    or (self.metrics_f_type[k] != NameToClass.BINARY)
                    and (self.n_class > NameToClass.NBINARY)
                )
            ]
        else:
            raise_PasoError(
                "parse_metrics; no type named: {} not in : {}".format(
                    self.model_type, NameToClass.__metrics__.keys()
                )
            )

        return self

    @pasoDecorators.TTWrapXy(array=False)
    def evaluate(self, X, y, **kwargs):
        """
        Parameters:
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.

        Returns:
            {'f1_score': value, 'logloss': value}

        """

        self._parse_metrics(y)

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        self.wrong_predicted_class = X[y != y_pred]

        self.metrics = {}
        for k in range(len(self.metrics_names)):

            if self.metrics_needs_proba[k]:
                y_predicted = y_pred_proba
            else:
                y_predicted = y_pred

            if (
                (self.metrics_f_type[k] == NameToClass.BINARY)
                and (self.n_class == NameToClass.NBINARY)
            ) or (
                (self.metrics_f_type[k] == NameToClass.MULTICLASS)
                and (self.n_class >= NameToClass.NMULTICLASS)
            ):
                # case binary classification and nclass == 2
                self.metrics = _merge_dicts(
                    self.metrics,
                    {
                        self.metrics_names[k]: self.metrics_f[k](
                            y, y_predicted, **self.metrics_f_kwargs[k]
                        )
                    },
                )
            else:
                pass

        self.evaluated = True
        return self.metrics

    @staticmethod
    def cross_validaters(self):
        # Todo:Rapids numpy
        """
        Parameters:
            None

        Returns:
            List of available models names.
        """
        return [k for k in NameToClass.__cross_validators__.keys()]

    @pasoDecorators.TTWrapXy(array=False)
    def cross_validate(self, X, y, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.
            cv_description_filepath:    str
        Returns: d
            dict: statistics of metrics
        """
        if self.cv_description_filepath == "":
            raise_PasoError(
                "Cross-validation requires description_filepath=\<fp\> keyword"
            )

        if self.kind == {}:
            raise_PasoError(
                "keyword kind=\<cross_validaters_name\> must be present at top level:{}:".format(
                    self.description_kwargs
                )
            )

        if self.kind_name not in NameToClass.__cross_validators__:
            raise_PasoError(
                "cross_validate; no operation named: {} __cross_validaters__ in cross_validaters;: {}".format(
                    self.kind_name, NameToClass.__cross_validators__.keys()
                )
            )

        self._parse_metrics(y)

        self.cross_validate_name = self.kind_name
        self.cross_validate_model = NameToClass.__cross_validators__[self.kind_name]
        self.cv = self.kind_name_kwargs["cv"]
        scores = self.cross_validate_model(
            self.model, X, y, scoring=self.metrics_scoring, **self.kind_name_kwargs
        )
        self.cv_metrics = sorted(scores.keys())
        self.cross_validate_model_type = self.type

        self.cross_validated = True

        return _stat_arrays_in_dict(scores)

    @pasoDecorators.TTWrapXy(array=False)
    def tune_hyperparameters(self, X, y, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            X:  pandas dataFrame
        Returns:
            self
        """
        if self.tune_description_filepath == "":
            raise_PasoError(
                "tune_hyperparameters-validation requires cv_description_filepath=\<fp\> keyword"
            )

        if self.kind == {}:
            raise_PasoError(
                "keyword kind=\<tune_hyperparameters_name\> must be present at top level:{}:".format(
                    self.description_kwargs
                )
            )

        if self.kind_name not in NameToClass._hp_optimizeers__:
            raise_PasoError(
                "hp_optimizers; no operation named: {} not in : {}".format(
                    self.kind_name, NameToClass.__hp_optimizers__.keys()
                )
            )
        else:
            self.hp_optimize_name = self.kind_name
            self.hp_optimize_model = NameToClass.__hp_optimizers__[self.kind_name](
                **self.kind_name_kwargs
            )
            self.hp_optimize_model.fit(X, y)
            self.hp_optimizee_model_type = self.type

        self.hp_optimized = True

        return self


########
