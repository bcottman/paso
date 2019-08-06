# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

#from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
# sklearn.metrics imports
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, log_loss, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import brier_score_loss

# paso imports
from paso.base import pasoModel, NameToClass
from paso.base import raise_PasoError
from paso.base import _array_to_string, pasoDecorators
from paso.base import _add_dicts
#from paso.base import _dict_value2
from loguru import logger

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

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):

        """
        Parameters:
            modelKey: (str) On
            verbose: (str) (default) True, logging off (``verbose=False``)

        Note:

        """
        super().__init__()
        self.inplace = False

    def learners(self):
        """
        Parameters:
            None

        Returns:
            List of available models names.
        """
        return list(NameToClass.__learners__.keys())

    @pasoDecorators.TrainWrapsklearn(array=False)
    def train(self, X, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            X:  pandas dataFrame
        Returns:
            self
        """

        # todo refactor using non-optional-kw
        # checks for non-optional keyword
        if self.learner not in NameToClass.__learners__:
            raise_PasoError(
                "TrainWrap: No learner named: {} not in learners;: {}".format(
                    self.learner, NameToClass.__learners_.keys()
                )
            )
        if self.cross_validation == None:
            raise_PasoError(
                "cross_validation not specified for learner: {}".format(self.learner)
            )
        if self.target == None:
            logger.warning(
                "target not specified through keyword call or ontological file for learner: {}".format(
                    self.target
                )
            )
        if self.target not in self.Xcolumns:
            raise_PasoError(
                "trainwrap: unknown target:{} in {}".format(self.target, self.Xcolumns)
            )
        # create instance of this particular learner
        if self.learner_name_kwargs == None:
            self.model = NameToClass.__learners__[self.learner]()
            logger.warning(
                "model_kwargs not specified for learner: {}".format(self.modelKey)
            )
        else:
            self.model = NameToClass.__learners__[self.learner](**self.kwargs)
            if self.verbose:
                logger.info(
                    "learner: {} with kwargs {}".format(self.learner, self.kwargs)
                )

        # X_train,y_train turned into arrays from dataframe, used in predict
        if self.target in self.Xcolumns:
            self.n_class = X[self.target].nunique()
            self.class_names = _array_to_string(X[self.target].unique())

        # todo support other cross-valid
        # todo  support named metrics
        self.y_train = X[self.target].values
        self.X_train = X[X.columns.difference([self.target])]
        self.clf = self.model  # start with base learner
        for i, cv_name in enumerate(self.cross_validation):
            if self.verbose:
                logger.info("cross_validation: {}".format(cv_name))

            cv_name_kwargs = self.cv_kwargs_list[i]
            if self.verbose:
                logger.info("    cv kwargs: {}".format(cv_name_kwargs))
            self.clf = NameToClass.__cross_validators__[cv_name](
                self.clf, **cv_name_kwargs
            )
            self.clf.fit(self.X_train, self.y_train)

        return self.clf

    @pasoDecorators.PredictWrap(array=False, narg=2)
    def predict(self, X, **kwargs):
        """
        Parameters:
            X_test:  pandas dataFrame #Todo:Dask numpy

            inplace: (CURRENTLY IGNORED)
                    False (boolean), replace 1st argument with resulting dataframe
                    True:  (boolean) ALWAYS False
        Returns:
            (DataFrame): predict of X
        """
        if self.measure:
            self.y_test = X[self.target].values
            self.X_test = X[X.columns.difference([self.target])]

        y_pred = self.clf.predict(self.X_test)

        if self.measure:
            self.wrong_predicted_class = X[X[self.target] != y_pred]  # df

        y_predp = self.clf.predict_proba(self.X_test)

        if self.type == "Classification":
            if self.measure:
                self.metrics = _add_dicts(
                    self.__metric_class(y_pred, self.y_test),
                    self.__metric_class_probability(y_predp, self.y_test),
                )

                if self.plot_confusion_matrix and self.verbose:
                    cm = confusion_matrix(self.y_test, y_pred)
                    self.metrics = _add_dicts(self.metrics, {"confusion_matrix": cm})
                    if self.verbose:
                        # Plot non-normalized confusion matrix
                        self._plot_confusion_matrix(
                            cm,
                            self.class_names,
                            normalize=False,
                            title="Confusion matrix, without normalization",
                        )
                        # Plot normalized confusion matrix
                        self._plot_confusion_matrix(
                            cm,
                            self.class_names,
                            normalize=True,
                            title="Normalized confusion matrix",
                        )
                        plt.show()
        # todo TBD
        elif self.type == "Regression":
            pass

        if self.measure:
            if self.verbose:
                logger.info(
                    "measures of error evaluated for model: {}".format(self.learner)
                )
        else:
            if self.verbose:
                logger.info(
                    "measures of error NOT evaluated for model:{}".format(self.learner)
                )

        return [y_pred, y_predp]

    def __metric_class(self, y_pred, y_act):
        """
            calculates classification metrics

            Parameters:
                y_pred:
                y_act:

            return: None

        """
        accuracy = accuracy_score(y_act, y_pred)
        precision = precision_score(y_act, y_pred, average="macro")
        recall = recall_score(y_act, y_pred, average="macro")
        f1 = f1_score(y_act, y_pred, average="macro")
        AOC = 0
        if self.n_class == 2:
            AOC = roc_auc_score(y_act, y_pred, average="macro")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "AOC": AOC,
        }

    def __metric_class_probability(self, y_pred_probability, y_act):
        """
            calculates classification metrics based on cass probability

            Parameters:
                y_pred_prob:
                y_act:

            return: None
        """
        logLoss = log_loss(y_act, y_pred_probability, eps=1e-15, normalize=True)
        if self.n_class == 2:
            # fix brier_loss
#            brier_loss = brier_score_loss(y_act, y_pred_probability)
            brier_loss = 0.0
            return {"brier_loss": brier_loss, "logloss": logLoss}
        else:
            return {"logloss": logLoss}

    def _plot_confusion_matrix(
        self, cm, class_names, normalize=False, title=None, cmap=plt.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
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
        rcParams["figure.figsize"] = (self.x_size, self.y_size)
        np.set_printoptions(precision=self.precision)
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


########
