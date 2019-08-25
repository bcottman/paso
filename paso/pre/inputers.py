import pandas as pd
from tqdm import tqdm
from pandas.util._validators import validate_bool_kwarg
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# paso imports
from paso.base import pasoFunction, raise_PasoError, _array_to_string
from paso.base import pasoDecorators, _check_non_optional_kw, _dict_value
from paso.base import _exists_as_dict_value, _kind_name_keys

from loguru import logger
import sys, os.path

# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#
def _inputer_exec(self, **kwargs):

    # must always be data = ' train' or no dataset =
    if self.dataset != "train" and ("train" not in kwargs):
        raise_PasoError(
            "dataset='{}'  not recognized: in  {} ".format(self.dataset, kwargs)
        )

    key = ["pre", "post"]
    if key[0] in kwargs and kwargs[key[0]] != None:
        for stmt in kwargs[key[0]]:
            exec(stmt)

    dfkey = "create-df"
    if dfkey in kwargs and kwargs[dfkey] != None:
        result = eval(kwargs[dfkey])

    if key[1] in kwargs and kwargs[key[1]] != "None":
        for stmt in kwargs[key[1]]:
            exec(stmt)

    return result


# def _create_path(kw, dictnary, directory_path, default):
#
#     if kw in dictnary:
#         return directory_path + _dict_value(dictnary, kw, default)
#     else:
#         return default


# todo refactor this mess
def _inputer_cvs(self, **kwargs):
    kw = "names"
    self.names = _dict_value(kwargs, kw, [])

    kw = "directory_path"
    self.directory_path = _dict_value(kwargs, kw, "")

    kw = "train"
    if kw in kwargs:
        self.train_path = self.directory_path + kwargs[kw]
        if os.path.exists(self.train_path):
            if self.names != []:
                train = pd.read_csv(self.train_path, names=self.names)
            elif self.names == []:
                train = pd.read_csv(self.train_path)
        else:
            raise_PasoError(
                "Inputer train dataset path does not exist: {} or there might not be a directory_path:{}".format(
                    self.train_path, self.directory_path
                )
            )

    kw = "url"
    if kw in kwargs:
        self.train_path = self.directory_path + kwargs[kw]
        if self.names != []:
            train = pd.read_csv(self.train_path, names=self.names)
        elif self.names == []:
            train = pd.read_csv(self.train_path)

    kw = "test"
    if kw in kwargs:
        self.test_path = self.directory_path + kwargs[kw]
        if os.path.exists(self.test_path):
            if self.names != []:
                test = pd.read_csv(self.test_path, names=self.names)
            elif self.names == []:
                test = pd.read_csv(self.test_path)
        else:
            raise_PasoError(
                "Inputer test dataset path does not exist: {}".format(self.test_path)
            )

    kw = "sampleSubmission"
    if kw in kwargs:
        self.sampleSubmission_path = self.directory_path + kwargs[kw]
        if os.path.exists(self.sampleSubmission_path):
            if self.names != []:
                sampleSubmission = pd.read_csv(
                    self.sampleSubmission_path, names=self.names
                )
            elif self.names == []:
                sampleSubmission = pd.read_csv(self.sampleSubmission_path)
        else:
            raise_PasoError(
                "Inputer sampleSubmission dataset path does not exist: {}".format(
                    self.test_path
                )
            )

    # no case in python
    if self.dataset == "train":
        return train
    elif self.dataset == "valid":
        return valid
    elif self.dataset == "test":
        return test
    elif self.dataset == "sampleSubmission":
        return sampleSubmission
    else:
        raise_PasoError("dataset not recognized: {} ".format(self.dataset))


def _inputer_xls(self, **kwargs):
    return None


def _inputer_xlsm(self, **kwargs):
    return None


def _inputer_text(self, **kwargs):
    return None


def _inputer_image2d(self, **kwargs):
    return None


def _inputer_image3d(self, **kwargs):
    return None


### Inputer
class Inputer(pasoFunction):
    """
    Input returns dataset.
    Tne metadata is the instance attibutesof Inputer prperties.

    Note:

    Warning:

    """

    __inputer__ = {
        "exec": _inputer_exec,
        "cvs": _inputer_cvs,
        "xls": _inputer_xls,
        "xlsm": _inputer_xlsm,
        "text": _inputer_text,
        "image2D": _inputer_image2d,
        "image3D": _inputer_image3d,
    }

    _datasets_avaialable_ = [
        "train",
        "valid",
        "test",
        "sampleSubmission",
        "directory_path",
    ]

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):

        """
        Parameters:
            filepath: (string)
            verbose: (boolean) (optiona) can be set. Default:True

        Note:

        """
        super().__init__()

    def inputers(self):
        """
        Parameters:
            None

        Returns:
            List of available inputer names.
        """
        return list(Inputer.__inputer__.keys())

    def datasets(self):
        """
        List type of files avaiable

        Parameters: None

        Returns: lists of datasets

        """
        return Inputer._datasets_avaialable_

    @pasoDecorators.TTWrap(array=False, narg=1)
    def transform(self, *args, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            ontology_filepath: path to yaml file containing dataset ontology

            Returns:
                dictnary
                    transformed X DataFrame
            Raises:

            Note:
        """

        # currently support only one inputer, very brittle parser
        kwa = "target"
        self.target = _dict_value(self.kind_name_kwargs, kwa, None)
        _check_non_optional_kw(
            kwa, "Inputer: needs target keyword. probably not set in ontological file."
        )
        # currently just  can only be in inputer/transformkwarg
        kwa = "dataset"
        self.dataset = _dict_value(kwargs, kwa, "train")

        # create instance of this particular learner
        # checks for non-optional keyword
        if self.kind_name not in Inputer.__inputer__:
            raise_PasoError(
                "transform; no format named: {} not in Inputers;: {}".format(
                    self.kind_name, Inputer.__inputer__.keys()
                )
            )
        else:
            return Inputer.__inputer__[self.kind_name](self, **self.kind_name_kwargs)


### Splitter
class Splitter(pasoFunction):
    """
    Input returns dataset.
    Tne metadata is the instance attibutesof Inputer prperties.

    Note:

    Warning:

    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):

        """
        Parameters:
            filepath: (string)
            verbose: (boolean) (optiona) can be set. Default:True

        Note:

        """
        super().__init__()
        self.inplace = False

    @pasoDecorators.TTWrap(array=False)
    def transform(self, X, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            target: dependent feature which is "target" of trainer

            Returns:
                    [train DF , test DF] SPLIT FROM X
            Raises:

            Note:
        """

        # c, can only be in kwarg in transform
        kwa = "target"
        self.target = _dict_value(kwargs, kwa, None)
        _check_non_optional_kw(
            self.target,
            "target key not specified in splitter:.target {}".format(kwargs),
        )
        if self.target not in self.Xcolumns:
            raise_PasoError(
                "splitter: {}: unknown target:{} in {}".format(
                    self, self.target, self.Xcolumns
                )
            )
        else:
            self.n_class = X[self.target].nunique()
            self.class_names = _array_to_string(X[self.target].unique())
            # note arrays
            y_train = X[self.target].values
            X_train = X[X.columns.difference([self.target])]
            # stratify =True them reset to y
            if "stratify" in self.kind_name_kwargs:
                self.kind_name_kwargs["stratify"] = y_train

            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, **self.kind_name_kwargs
            )

            X_train = pd.DataFrame(
                X_train, columns=[item for item in self.Xcolumns if item != self.target]
            )
            X_train[self.target] = y_train

            X_test = pd.DataFrame(
                X_test, columns=[item for item in self.Xcolumns if item != self.target]
            )
            X_test[self.target] = y_test

            return X_train, X_test


###
