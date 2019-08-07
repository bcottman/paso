import pandas as pd
from tqdm import tqdm
from pandas.util._validators import validate_bool_kwarg
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# paso imports
from paso.base import pasoFunction, raise_PasoError, _array_to_string
from paso.base import pasoDecorators, _check_non_optional_kw, _dict_value

from loguru import logger
import sys, os.path

# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#
def _inputer_exec(self, dictnary):

    # must always be datae = ' train' or no dataset =
    if self.dataset != "train" and ("train" not in dictnary):
        raise_PasoError("dataset not recognized: {} ".format(self.dataset))

    key = ["pre", "post"]
    if key[0] in dictnary and dictnary[key[0]] != None:
        for stmt in dictnary[key[0]]:
            exec(stmt)

    dfkey = "create-df"
    if dfkey in dictnary and dictnary[dfkey] != None:
        result = eval(dictnary[dfkey])

    if key[1] in dictnary and dictnary[key[1]] != "None":
        for stmt in dictnary[key[1]]:
            exec(stmt)

    return result


def _create_path(kw, dictnary, directory_path, default):

    if kw in dictnary:
        return directory_path + _dict_value(dictnary, kw, default)
    else:
        return default


# todo refactor this mess
def _inputer_cvs(self, dictnary):
    kw = "names"
    self.names = _dict_value(dictnary, kw, [])

    kw = "directory_path"
    self.directory_path = _dict_value(dictnary, kw, "")

    kw = "train"
    if kw in dictnary:
        self.train_path =self.directory_path + dictnary[kw]
        if os.path.exists(self.train_path):
            if self.names != []:
                train = pd.read_csv(self.train_path, names=self.names)
            elif self.names == []:
                train = pd.read_csv(self.train_path)
        else:
            raise_PasoError(
                "Inputer train path does not exist: {} or there might not be a directory_path:{}".format(
                    self.train_path, self.directory_path
                )
            )

    kw = "url"
    if kw in dictnary:
        self.train_path = self.directory_path + dictnary[kw]
        if self.names != []:
            train = pd.read_csv(self.train_path, names=self.names)
        elif self.names == []:
            train = pd.read_csv(self.train_path)

    kw = "test"
    if kw in dictnary:
        self.test_path = self.directory_path + dictnary[kw]
        if os.path.exists(self.test_path):
            if self.names != []:
                test = pd.read_csv(self.test_path, names=self.names)
            elif self.names == []:
                test = pd.read_csv(self.test_path)
        else:
            raise_PasoError(
                "Inputer test path does not exist: {}".format(self.test_path)
            )

    kw = "sampleSubmission"
    if kw in dictnary:
        self.sampleSubmission_path = self.directory_path + dictnary[kw]
        if os.path.exists(self.sampleSubmission_path):
            if self.names != []:
                sampleSubmission = pd.read_csv(
                    self.sampleSubmission_path, names=self.names
                )
            elif self.names == []:
                sampleSubmission = pd.read_csv(self.sampleSubmission_path)
        else:
            raise_PasoError(
                "Inputer sampleSubmission path does not exist: {}".format(
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


def _inputer_xls(self, dict):
    return None


def _inputer_xlsm(self, dict):
    return None


def _inputer_text(self, dict):
    return None


def _inputer_image2d(self, dict):
    return None


def _inputer_image3d(self, dictnary):
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

    @pasoDecorators.TransformWrapnarg(narg=1)
    def datasets(self):
        """
        List type of files avaiable

        Parameters: None

        Returns: lists of datasets

        """

        if _check_non_optional_kw(
            self.format, msg="Inputer:transform bad format: {}".format(self.format)
        ):
            if self.format in Inputer.__inputer__:
                file_name_list = Inputer._datasets_avaialable_
                _check_non_optional_kw(
                    self.formatDict,
                    msg="Inputer:transform bad format not foumd: {}".format(
                        self.format
                    ),
                )
                result = [name for name in file_name_list if name in self.formatDict]
                if result == []:
                    result = ["train"]

        return result

    @pasoDecorators.TransformWrapnarg(narg=1)
    def transform(self, **kwargs):
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

        # check keywords in passes argument stream
        # non-optional kw are initiated with None

        if _check_non_optional_kw(
            self.format, msg="Inputer:transform bad format: {}".format(self.format)
        ):
            if self.format in Inputer.__inputer__:
                self.input_fun = Inputer.__inputer__[self.format]
                _check_non_optional_kw(
                    self.formatDict,
                    msg="Inputer:transform bad format not foumd: {}".format(
                        self.format
                    ),
                )

        return self.input_fun(self, self.formatDict)


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

    @pasoDecorators.TransformWrapnarg(array=False, narg=2)
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

        # check keywords in passes argument stream
        # non-optional kw are initiated with None

        if self.target == None:
            raise_PasoError(
                "target not specified through keyword call or ontological file for splitter: {}".format(
                    self
                )
            )

        if self.kwargs == None:
            raise_PasoError(
                "train_test_split_kwargs not specified for splitter: {}".format(self)
            )

        if self.target not in self.Xcolumns:
            raise_PasoError(
                "splitter: {}: unknown target:{} in {}".format(
                    self, self.target, self.Xcolumns
                )
            )

        if _check_non_optional_kw(
            self.target,
            msg="Splitter:transform target= non-optional: {}".format(self.target),
        ):
            if self.target in self.Xcolumns:
                self.n_class = X[self.target].nunique()
                self.class_names = _array_to_string(X[self.target].unique())
                # note arrays
                y_train = X[self.target].values
                X_train = X[X.columns.difference([self.target])]
                # stratify =True them reset to y
                if "stratify" in self.kwargs and self.kwargs["stratify"]:
                    self.kwargs["stratify"] = y_train

            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

            X_train[self.target] = y_train
            X_test[self.target] = y_test

        return X_train, X_test


###
