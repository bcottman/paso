# !/usr/bin/env python
# -*- coding: utf-8 -*-
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
from paso.base import _array_to_string, pasoDecorators, _kind_name_keys
from paso.base import _add_dicts, _exists_Attribute


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
        self.metrics_scoring = {
            k: make_scorer(v[0], **v[1])
            for k, v in NameToClass.__metrics__["Classification"].items()
        }
        self.metrics_names = [
            k for k in NameToClass.__metrics__["Classification"].keys()
        ]
        self.cross_validated = False
        self.tuned_parameters = False

    def learners(self):
        """
        Parameters:
            None

        Returns:
            List of available models names.
        """
        return list(NameToClass.__learners__.keys())

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
                "Train; no operation named: {} not in learners;: {}".format(
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

    @pasoDecorators.TTWrapXy(array=False)
    def evaluate(self, X, y, **kwargs):
        """
        Parameters:
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.

        Returns:
            {'f1_score': value, 'logloss': value}

        """
        # assured by predictwrap that train has been called already
        # _check_non_optional_kw(
        #     self.measure, "measure kw must be present in call to predict"
        # )
        if self.model_type == "Classification":
            self.n_class = len(np.unique(y))
            self.class_names = _array_to_string(np.unique(y))
            y_pred = self.predict(X)
            y_pred_proba = self.predict_proba(X)
            self.wrong_predicted_class = X[y != y_pred]  # df
            self.metrics = _add_dicts(
                self._metric_classification(y, y_pred),
                self._metric_classification_probability(y, y_pred_proba),
            )
            cm = confusion_matrix(y, y_pred)
            self.metrics = _add_dicts(self.metrics, {"confusion_matrix": cm})

            # todo TBD
        elif self.model_type == "Regression":
            pass

        self.evaluated = True
        return self.metrics

    def _metric_classification(self, y_act, y_pred):
        def __metric_classification(self, y_act, y_pred):
            """
                calculates classification metrics

                Parameters:
                    y_act: (numpy vector ) target or dependent feature of dataset.
                    y_pred: (numpy vector of ints)  predicted target.

                return: dict {
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "AOC": AOC
                                }

            """

        accuracy = accuracy_score(y_act, y_pred)
        precision = precision_score(y_act, y_pred, average="macro")
        recall = recall_score(y_act, y_pred, average="macro")
        f1 = f1_score(y_act, y_pred, average="macro")
        AOC = "must be binary class"
        if self.n_class == 2:
            AOC = roc_auc_score(y_act, y_pred, average="macro")
            # todo:%use NameClass list
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "AOC": AOC,
        }

    def _metric_classification_probability(self, y_act, y_pred_probability):
        """
            calculates classification metrics based on class probability

            Parameters:
                y_act: (numpy vector ) target or dependent feature of dataset.
                y_pred: (numpy vector of ints)  predicted target.

            return: (dict) {"logloss": logLoss}
        """
        logLoss = log_loss(y_act, y_pred_probability, eps=1e-15, normalize=True)
        if self.n_class == 2:
            # fix brier_loss
            #            brier_loss = brier_score_loss(y_act, y_pred_probability)
            brier_loss = 0.0
            return {"brier_loss": brier_loss, "logloss": logLoss}
        else:
            # todo:%use NameClass list
            return {"logloss": logLoss}

    def plot_confusion_matrix(self, cm, normalize=False):
        # Plot non-normalized confusion matrix
        if normalize:
            title = (
                "Confusion matrix, with normalization"
                + self.model_name
                + "  "
                + self.model_type
            )
        else:
            title = "Confusion matrix " + self.model_name + "  " + self.model_type

        self._plot_confusion_matrix(
            cm, self.class_names, normalize=normalize, title=title
        )

        plt.show()

    def _plot_confusion_matrix(
        self, cm, class_names, normalize=False, title=None, cmap=plt.cm.Blues
    ):
        """
        This function gprahically plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = "Normalized confusion matrix"
            else:
                title = "Confusion matrix, without normalization"

        # Compute confusion matrix
        # Only use the labels that appear in the data

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            if self.verbose:
                logger.info("Normalized confusion matrix")
        else:
            if self.verbose:
                logger.info("Confusion matrix, without normalization")

        fig, ax = plt.subplots()

        #        rcParams["figure.figsize"] = (self.x_size, self.y_size)
        rcParams["figure.figsize"] = (6.0, 6.0)
        #        np.set_printoptions(precision=self.precision)
        np.set_printoptions(precision=3)

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = ".3f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        return ax

    @pasoDecorators.TTWrapXy(array=False)
    def cross_validaors(self):
        # Todo:Rapids numpy
        """
        Parameters:
            None

        Returns:
            List of available models names.
        """
        return [k for k in NameToClass.__cross_validators__.keys()]

    @pasoDecorators.TTWrapXy(array=False, kwarg_description_filepath_parse=True)
    def cross_validate(self, X, y, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.
        Returns: d
            dict: statistics of metrics
        """
        # currently support only one learner, very brittle parser
        if self.kind == {}:
            raise_PasoError(
                "keyword kind must be present at top level:{}:".format(
                    self.description_kwargs
                )
            )

        if self.kind_name not in NameToClass.__cross_validators__:
            raise_PasoError(
                "cross_validate; no operation named: {} __cross_validaters__ in cross_validaters;: {}".format(
                    self.kind_name, NameToClass.__cross_validators__.keys()
                )
            )
        else:
            self.cross_validate_name = self.kind_name
            self.cross_validate_model = NameToClass.__cross_validators__[self.kind_name]
            self.cv = self.kind_name_kwargs["cv"]
            # only binary
            if len(np.unique(y)) > 2:
                del self.metrics_scoring["AOC"]
            scores = self.cross_validate_model(
                self.model, X, y, scoring=self.metrics_scoring, **self.kind_name_kwargs
            )
            self.cv_metrics = sorted(scores.keys())
            self.cross_validate_model_type = self.type

        self.cross_validated = True

        return _stat_arrays_in_dict(scores)

    @pasoDecorators.TTWrapXy(array=False)
    def hp_optimize(self, X, y, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            X:  pandas dataFrame
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
