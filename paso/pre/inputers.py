import pandas as pd
from tqdm import tqdm
from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")

# paso imports
from paso.base import pasoFunction, PasoError, raise_PasoError
from paso.base import pasoDecorators, _check_non_optional_kw
from loguru import logger
import sys

# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#
def _inputer_exec(self, dict):

    key = ['pre', 'post']
    if key[0] in dict and dict[key[0]] != None:
        logger.debug(dict[key[0]])
        for stmt in dict[key[0]]:
            exec(stmt)

    dfkey = 'create-df'
    if dfkey in dict and dict[dfkey] != None:
        logger.debug(dict[dfkey])
        self.f_x = eval(dict[dfkey])

    if key[1] in dict and dict[key[1]] != 'None':
        for stmt in dict[key[1]]:
            exec(stmt)

    return self.f_x

def _inputer_cvs(self, dict):
    return None

def _inputer_xls(self, dict):
    return None

def _inputer_xlsm(self, dict):
    return None

def _inputer_text(self, dict):
    return None

def _inputer_image2d(self, dict):
    return None

def _inputer_image3d(self, dict):
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
        "exec": _inputer_exec
        ,"cvs": _inputer_cvs
        ,"xls": _inputer_xls
        ,"xlsm": _inputer_xlsm
        ,"text": _inputer_text
        ,"image2D": _inputer_image2d
        ,"image3D": _inputer_image3d
    }

    def __init__(self):

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
    def transform(self, ontology_filepath=""):
        # Todo:Rapids numpy
        """
        Parameters:
            ontology_filepath: path to yaml file containing dataset ontology

            Returns:
                dict
                    transformed X DataFrame
            Raises:

            Note:
        """

        # check keywords in passes argument stream
        # check keywords in passes argument stream
        # non-optional kw are initiated with None

        if _check_non_optional_kw(self.format
                ,msg='Inputer:transform bad format: {}'.format(self.format)):
            if self.format in Inputer.__inputer__:
                self.input_fun = Inputer.__inputer__[self.format]
            if _check_non_optional_kw(self.formatDict
                    ,msg='Inputer:transform bad format not foumd: {}'.format(self.format)):
                self.f_x = self.input_fun(self, self.formatDict)

        return self.f_x



###
