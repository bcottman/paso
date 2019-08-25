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

# from sklearn.metrics import brier_score_loss

# paso imports
from paso.base import pasoModel, NameToClass, _check_non_optional_kw
from paso.base import raise_PasoError, _exists_as_dict_value, _dict_value
from paso.base import _array_to_string, pasoDecorators, _kind_name_keys
from paso.base import _add_dicts, _exists_Attribute


# from paso.base import _dict_value2
from loguru import logger

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
# class Learner
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

    def learners(self):
        """
        Parameters:
            None

        Returns:
            List of available models names.
        """
        return list(NameToClass.__learners__.keys())

    @pasoDecorators.TTWrap(array=False)
    def train(self, X, *args, **kwargs):
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
        if _exists_Attribute(self, "target"):
            if self.target in self.Xcolumns:
                self.y_train = X[self.target].values
                self.X_train = X[X.columns.difference([self.target])].values
            # no target set,
            else:
                raise_PasoError(
                    "learner:fit: unknown target:{} in {} for learner: {}".format(
                        self.target, self.Xcolumns, self.kind_name
                    )
                )
        else:
            raise_PasoError(
                "learner:train: target: in must be a keyword in call to train: {}".format(
                    kwargs
                )
            )

        # list(self.kind.keys())[0]
        # self.kind_name_key = _exists_as_dict_value(self.kind, self.kind_name)
        # # Thus order keyword must be given even if []
        # kind_name_kwargs = _kind_name_keys(
        #     self, self.kind_name_key, verbose=self.verbose
        # )

        # create instance of this particular learner
        # checks for non-optional keyword
        if self.kind_name not in NameToClass.__learners__:
            raise_PasoError(
                "Train; no operation named: {} not in learners;: {}".format(
                    self.kind_name, NameToClass.__learners__.keys()
                )
            )
        else:
            self.model = NameToClass.__learners__[self.kind_name](
                **self.kind_name_kwargs
            )
            self.model.fit(self.X_train, self.y_train)
            self.clf_type = self.type

        # operation on this class by keyword order
        kwa = "order"
        order_ops = _dict_value(
            self.kind_name_keys, kwa, []
        )  # should be a lis or error, no constant
        for id_ops, ops_name in enumerate(order_ops):
            ops_name_key = _exists_as_dict_value(self.kind_name_keys, ops_name)
            ops_name_kwargs = _kind_name_keys(self, ops_name_key, verbose=self.verbose)
            if self.verbose:
                logger.info(
                    "\n clf-ops-name:{}\n kwargs: {}".format(ops_name, ops_name_kwargs)
                )
            self.model = NameToClass.__learners__[ops_name](
                self.model, **ops_name_kwargs
            )
            self.model.fit(self.X_train, self.y_train)

        return self

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
        # assured by predictwrap that train has been called already
        _check_non_optional_kw(
            self.measure, "measure kw must be present in call to predict"
        )
        if self.clf_type == "Classification":
            if self.measure:
                self.n_class = X[self.target].nunique()
                self.class_names = _array_to_string(X[self.target].unique())
                self.y_test = X[self.target].values
                self.X_test = X[X.columns.difference([self.target])].values
                y_pred = self.model.predict(self.X_test)
                self.wrong_predicted_class = X[X[self.target] != y_pred]  # df
                y_predp = self.model.predict_proba(self.X_test)
                logger.debug("type:{}.".format(self.type))
                if self.clf_type == "Classification":
                    self.metrics = _add_dicts(
                        self.__metric_class(y_pred, self.y_test),
                        self.__metric_class_probability(y_predp, self.y_test),
                    )
                    if self.verbose:
                        cm = confusion_matrix(self.y_test, y_pred)
                        self.metrics = _add_dicts(
                            self.metrics, {"confusion_matrix": cm}
                        )
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
                else:
                    if self.verbose:
                        logger.info(
                            "measures of error NOT evaluated for model:{}".format(
                                self.kind_name
                            )
                        )

            # todo TBD
            elif self.clf_type == "Regression":
                pass

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


########
