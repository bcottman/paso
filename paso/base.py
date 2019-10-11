#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

""
# todo:  enable checkpoint
# todo: support RAPID in parm file
# todo: port to swift
# not neeed with checkpoints and a notebook for every pipelin

from abc import ABC
from loguru import logger
import pydot_ng as pydot
from IPython.display import Image, display
import sys, os.path
import yaml
from attrdict import AttrDict
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.util._validators import validate_bool_kwarg
import timeit, math
from tqdm import tqdm
import multiprocessing as mp
from numba import jit

#
import warnings

warnings.filterwarnings("ignore")


class PasoError(Exception):
    pass


def raise_PasoError(msg):
    logger.error(msg)
    raise PasoError(msg)


def Xy_to_DataFrame(X, y, target):
    """
        Tranform X(DataFrame and y (numpy vector) into one DataFame X

        Note: Memory is saved as y added to X dataFrame in place.

    :param X: DataFrame
    :param y: numpy vector
    :param target: sring
    :return: X
    """
    X[target] = y
    return None


def DataFrame_to_Xy(X, target):
    y = X[target].values
    X = X[X.columns.difference([target])]
    return X, y


def _must_be_list_tuple_int(attribute):
    if isinstance(attribute, (tuple, list)):
        return attribute
    elif isinstance(attribute, (str, int, float)):
        return attribute
    else:
        raise_PasoError(
            "{} must be tiple,list, str, float, or int. unsupported type: {}".format(
                attribute, type(attribute)
            )
        )


@jit
def _divide_dict(d, n):
    return {k: d[k] / n for k in d}


# @jit
def _mean_arrays_in_dict(d):
    return {k: np.mean(d[k]) for k in d}


# @jit
def _median_arrays_in_dict(d):
    return {k: np.median(d[k]) for k in d}


# @jit
def _std_arrays_in_dict(d):
    return {k: np.std(d[k]) for k in d}


# @jit
def _var_arrays_in_dict(d):
    return {k: np.var(d[k]) for k in d}


# @jit
def _stat_arrays_in_dict(d):
    return {
        "mean": _mean_arrays_in_dict(d),
        "median": _median_arrays_in_dict(d),
        "std": _std_arrays_in_dict(d),
        "var": _var_arrays_in_dict(d),
    }


def _add_dicts(d1, d2):
    for k in d1.keys():
        if k in d2.keys():
            d1[k] = d1[k] + d2[k]
    return {**d2, **d1}


def _array_to_string(array):
    return [str(item) for item in array]


def is_Series(X):
    """
    Parameters:
        X (any type)

    Returns:
        True (boolean) if DataFrame Series type.
        False (boolean) otherwise
    """
    return isinstance(X, pd.core.series.Series)


def is_DataFrame(X):
    """
    Parameters:
        X (any type)

    Returns:
        True (boolean) if DataFrame type.
        False (boolean) otherwise
    """
    return isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series))


def _new_feature_names(X, names):
    """
    Change feature nanmes of a dataframe to names.

    Parameters:
        X: Dataframe
        names: list,str

    Returns: X

    """
    # X is inplace because only changing column names
    if names == []:
        return X
    c = list(X.columns)
    if isinstance(names, (list, tuple)):
        c[0 : len(labels)] = labels
    else:
        c[0:1] = [labels]
    X.columns = c
    return X


# must be dataFrame or series
def _Check_is_DataFrame(X):
    if is_DataFrame(X):
        return True
    else:
        raise_PasoError(
            "TransformWrap:Xarg must be if type DataFrame. Was type:{}".format(
                str(type(X))
            )
        )


def _Check_No_NA_F_Values(df, feature):
    if not df[feature].isna().any():
        return True
    else:
        raise_PasoError("Passed dataset, DataFrame, contained NA")


def _Check_No_NA_Series_Values(ds):
    if not ds.isna().any():
        return True
    else:
        raise_PasoError("Passed dataset, DataFrame, contained NA")


def _Check_No_NA_Values(df):
    for feature in df.columns:
        if _Check_No_NA_F_Values(df, feature):
            pass
        else:
            pass


def set_modelDict_value(v, at):
    """
    replaced by _dict_value
    """
    if at not in objecty.modelDict.keys():
        objecty.modelDict[at] = v


def _exists_as_dict_value(dictnary, key):
    """
    used to variable to dict-value or default.
    if key in dict return key-value
    else return default.

    """
    if key in dictnary:
        if isinstance(dictnary[key], dict):
            return dictnary[key]
        else:
            raise_PasoError("{} is NOt of type dictionary".format(key))
    else:
        logger.warning("{} is no in dictionary:({})".format(key, dictnary))
        return {}


def _dict_value(dictnary, key, default):
    """
    used to variable to dict-value or default.
    if key in dict return key-value
    else return default.

    """
    if key in dictnary:
        return dictnary[key]
    else:
        return default


def _dict_value2(dictnary, fdictnary, key, default):
    """
    used to variable to dict or fdict (2nd dict) value or default.
    if key in dict or fdict return key-value
    else return default.

    if in both, fdict is given precedent

    """
    result = default
    if key in dictnary:
        result = dictnary[key]
    if key in fdictnary:
        result = fdictnary[key]  # precedence to ontoloical file
    return result


def _check_non_optional_kw(kw, msg):
    """
    Raise PasoError and halt if non-optinal keyword not present.
    kw must be in namespace and set to None.

    """
    if kw == None:
        raise_PasoError(msg)
    else:
        return True


def _exists_Attribute(objecty, attribute_string):
    return hasattr(objecty, attribute_string)


def _exists_key(dictnary, key, error=True):
    if key in dictnary:
        return dictnary[key]
    else:
        if error:
            raise_PasoError(
                "learner:fit/predict:{} not specified through keyword call or description file for: {}".format(
                    attribute, self.name
                )
            )
        else:
            logger.warning(
                "{} not specified. very likely NOT error): {}".format(
                    attribute, self.name
                )
            )
            return False


def _kind_name_keys(o, kind_name_keys, verbose=True):
    """
        Accomplish parsing of a named class/ genus. kind_name(s) key/value pairs.
        This group is the dict of the ``kind`` key.
        ``#kind-<name>-keys``:
        genus: learner                                #[optional]
        type: Classification
        description: "Generic RF for classification"  #[optional]
        kwargs:
          n_estimators: 100
          n_jobs: -1
          criterion: gini   #entropy
          max_features: sqrt
          max_depth: 50
        order: [ops0, ops1, ...ops-n]

    """
    kwa = "description"
    o.description = _dict_value(kind_name_keys, kwa, None)

    kwa = "genus"
    o.genus = _dict_value(kind_name_keys, kwa, None)

    kwa = "type"
    o.type = _dict_value(kind_name_keys, kwa, None)

    # even if blank must be there
    kwa = "kwargs"
    o.kind_name_kwargs = _exists_as_dict_value(kind_name_keys, kwa)

    o.kind_name_keys = kind_name_keys

    return o.kind_name_kwargs


from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.linear_model import LinearRegression, Lasso, LassoLars
from sklearn.linear_model import LassoLars, ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor, SGDClassifier

from sklearn.linear_model import HuberRegressor


from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from hpsklearn import HyperoptEstimator

# sklearn.metrics imports
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, log_loss, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

### NameToClass class
class NameToClass:
    """
        map sklearner name to class stub
    """

    __learners__ = {
        "LGBMClassifier": LGBMClassifier,
        "SGDClassifier": SGDClassifier,
        "RandomForest": RandomForestClassifier,  # ......estimator
        "XGBClassifier": XGBClassifier,
        "ARDRegression": ARDRegression,
        "BayesianRidge": BayesianRidge,
        "ElasticNet": ElasticNet,
        "HuberRegressor": HuberRegressor,
        "Lasso": Lasso,
        "LassoLars": LassoLars,
        "LGBCRegression": LGBMRegressor,
        "LinearRegression": LinearRegression,
        "LogisticRegression": LogisticRegression,
        "Ridge": Ridge,
        "SGDRegressor": SGDRegressor,
        "XGBRegression": XGBRegressor,
    }

    __cross_validators__ = {
        "cross_validate": cross_validate,
        "BaggingClassifier": BaggingClassifier,  # ......cross-validators,
        "CalibratedClassifierCV": CalibratedClassifierCV,
    }

    __hyper_parameter_tuners__ = {"Hyperopt": HyperoptEstimator}

    __metrics__ = {
        #   Type            metric_name metric_func kwargs              greater_is_better  needs_proba
        "Classification": {
            "AOC": [roc_auc_score, {"greater_is_better": True, "needs_proba": False}],
            "accuracy": [
                accuracy_score,
                {"greater_is_better": True, "needs_proba": False},
            ],
            "f1_score": [
                f1_score,
                {"greater_is_better": True, "needs_proba": False, "average": "micro"},
            ],
            "precision": [
                precision_score,
                {"greater_is_better": True, "needs_proba": False, "average": "micro"},
            ],
            "recall": [
                recall_score,
                {"greater_is_better": True, "needs_proba": False, "average": "micro"},
            ],
            "logloss": [log_loss, {"greater_is_better": False, "needs_proba": True}],
        },
        "Regression": {},
        "Clustering": {},
        "Dimension_Reduction": {},
    }


def _narg_boiler_plate(objecty, narg, _Check_No_NAs, fun, array, *args, **kwargs):
    """ hidden postamble for transform, train wrap"""
    if narg == 1:
        result = fun(objecty, **kwargs)
    elif narg == 2:
        if objecty.inplace:
            X = args[1]
        else:
            X = args[1].copy()
        _Check_is_DataFrame(X)
        objecty.Xcolumns = X.columns
        if _Check_No_NAs:
            _Check_No_NA_Values(X)

        # if len(args) == 2:
        #     args = []  # klug for AugmentBy_class
        # pre
        if array:
            result = fun(objecty, X.to_numpy(), **kwargs)
        else:
            result = fun(objecty, X, **kwargs)
    elif narg == 3:
        if objecty.inplace:
            X = args[1]
            y = args[2]
        else:
            X = args[1].copy()
            y = args[2].copy()
        _Check_is_DataFrame(X)
        objecty.Xcolumns = X.columns
        if _Check_No_NAs:
            _Check_No_NA_Values(X)
        if array:
            result = fun(objecty, X.to_numpy(), y, **kwargs)
        else:
            result = fun(objecty, X, y, **kwargs)
    else:
        raise_PasoError(" _;  X ; X,y only currently supported in TTWrap")
    return result


def _kwargs_parse(*args, **kwargs):
    """ generic preamble for ontological ok kw for transform, train wrap
        description pairs:
            project: HPKinetics/paso #[optional]
            verbose: True  #[optional]
            inplace: True #[optional]

            kind:
                <class name>:
        kwarg pairs:
            target: (learner only)
    """
    objecty = args[0]

    kwa = "description_filepath"  # set in __init__
    if _exists_Attribute(objecty, kwa):
        # not supposed to be in transform kw
        objecty.description_kwargs = Param(
            filepath=objecty.description_filepath
        ).parameters_D

    # if kwa in kwargs:
    #     raise_PasoError(
    #         "description_filepath not given in instance creation. Transform/Train  wrong place."
    #     )

    # if in  description file then keyword will have precedence
    # check keywords in passes argument stream
    # non-optional kw default as None
    # head of description file
    kwa = "project"
    objecty.project = _dict_value2(kwargs, objecty.description_kwargs, kwa, None)

    kwa = "inplace"
    objecty.inplace = _dict_value2(kwargs, objecty.description_kwargs, kwa, True)
    validate_bool_kwarg(objecty.inplace, kwa)

    kwa = "verbose"
    objecty.verbose = _dict_value2(kwargs, objecty.description_kwargs, kwa, True)
    validate_bool_kwarg(objecty.verbose, kwa)

    kwa = "dataset_name"  # opional
    objecty.verbose = _dict_value2(kwargs, objecty.description_kwargs, kwa, "")

    if objecty.verbose:
        logger.info(
            "Loaded description_filepath file:{} ".format(objecty.description_filepath)
        )

    kwa = "kind"
    objecty.kind = _exists_as_dict_value(
        objecty.description_kwargs, kwa
    )  # ui not- error

    # check keywords in passes argument stream
    # non-optional kw are initiated with None

    # currently support only one inputer, very brittle parser
    if objecty.kind == {}:
        raise_PasoError(
            "keyword kind must be present at top level:{}:\n or indented key-vakuee pairs do not follow kind".format(
                objecty.description_kwargs
            )
        )
    # currently 1st and only as value as dict
    objecty.kind_name = list(objecty.kind.keys())[0]
    objecty.kind_name_keys = _exists_as_dict_value(objecty.kind, objecty.kind_name)
    # Thus order keyword must be given even if []
    objecty.kind_name_kwargs = _kind_name_keys(
        objecty, objecty.kind_name_keys, verbose=objecty.verbose
    )
    if objecty.verbose:
        logger.info(
            "\n kind-name:{}\n description:{}\n genus: {}\n type: {}\n kwargs: {} \n".format(
                objecty.kind_name,
                objecty.description,
                objecty.genus,
                objecty.type,
                objecty.kind_name_kwargs,
            )
        )

    # currently just learner/training models, can only be in kwarg in train
    # kwa = "measure"
    # objecty.measure = _dict_value2(kwargs, objecty.description_kwargs, kwa, True)
    # validate_bool_kwarg(objecty.measure, kwa)

    # currently just learner/training models, can only be in kwarg in train
    kwa = "target"
    objecty.target = _dict_value2(kwargs, objecty.description_kwargs, kwa, None)

    return objecty


def _kwarg_description_filepath_parse(*args, **kwargs):
    """ generic preamble for description ok kw for transform, train wrap
        description pairs:
            project: HPKinetics/paso #[optional]
            verbose: True  #[optional]
            inplace: True #[optional]

            kind:
                <class name>:
        kwarg pairs:
            target: (learner only)
    """
    objecty = args[0]

    default = ""
    objecty.description_filepath = _dict_value(kwargs, "description_filepath", default)
    if objecty.description_filepath == default:
        logger.warning(
            "No description_filepath instance:, args: {},kwargs: {}".format(
                args, kwargs
            )
        )

    return True


### pasoDecorators class
class pasoDecorators:
    def InitWrap():
        """
            Hide most of the paso machinery, so that developer focuses on their function or method.

            Parameters: None

                verbose

            return:
                What the decorated function returns.

        """

        def decorator(fun):
            # i suppose i could of done @wraos for self, but this works
            def wrapper(*args, **kwargs):

                objecty = args[0]
                # any class can have keyord verbose = (booean)

                default = ""
                objecty.description_filepath = _dict_value(
                    kwargs, "description_filepath", default
                )
                if objecty.description_filepath == default:
                    logger.warning(
                        "No description_filepath instance:, args: {},kwargs: {}".format(
                            args, kwargs
                        )
                    )

                fun(objecty, **kwargs)

            return wrapper

        return decorator

    def TTWrapNoArg(array=False, _Check_No_NAs=True):
        """
            Hide most of the paso machinery, so that developer focuses on their function or method.

            Parameters:
                array: (boolean) False
                    Pass a Pandas dataframe (False) or numpy array=True.  Mainly for compatibility
                    with scikit which requires arrays.

            Parm/wrap Class Instance attibutes: these are attibutes of objecty.fun (sef.x) set in this wrapper, if present in Parameter file

        """

        def decorator(fun):
            def wrapper(*args, **kwargs):
                objecty = _kwargs_parse(*args, **kwargs)
                result = _narg_boiler_plate(
                    objecty, 1, _Check_No_NAs, fun, array, *args, **kwargs
                )
                # post
                objecty.transformed = True
                objecty.trained = True
                return result

            return wrapper

        return decorator

    def TTWrap(array=False, _Check_No_NAs=True):
        """
            Hide most of the paso machinery, so that developer focuses on their function or method.

            Parameters:
                array: (boolean) False
                    Pass a Pandas dataframe (False) or numpy array=True.  Mainly for compatibility
                    with scikit which requires arrays.

            Parm/wrap Class Instance attibutes: these are attibutes of objecty.fun (sef.x) set in this wrapper, if present in Parameter file

        """

        def decorator(fun):
            def wrapper(*args, **kwargs):
                objecty = _kwargs_parse(*args, **kwargs)
                result = _narg_boiler_plate(
                    objecty, 2, _Check_No_NAs, fun, array, *args, **kwargs
                )
                # post
                objecty.transformed = True
                objecty.trained = True
                return result

            return wrapper

        return decorator

    def TTWrapXy(
        array=False, _Check_No_NAs=True, kwarg_description_filepath_parse=False
    ):
        """
            Hide most of the paso machinery, so that developer focuses on their function or method.

            Parameters:
                array: (boolean) False
                    Pass a Pandas dataframe (False) or numpy array=True.  Mainly for compatibility
                    with scikit which requires arrays.

            Parm/wrap Class Instance attibutes: these are attibutes of objecty.fun (sef.x) set in this wrapper, if present in Parameter file

        """

        def decorator(fun):
            def wrapper(*args, **kwargs):
                if kwarg_description_filepath_parse:
                    _kwarg_description_filepath_parse(*args, **kwargs)
                objecty = _kwargs_parse(*args, **kwargs)
                result = _narg_boiler_plate(
                    objecty, 3, _Check_No_NAs, fun, array, *args, **kwargs
                )
                # post
                objecty.transformed = True
                objecty.trained = True
                return result

            return wrapper

        return decorator

    def PredictWrap(array=False, narg=2):
        """
            Hide most of the paso machinery, so that developer focuses on their function or method.

            Parameters:
                array: )(boolean) False
                    Pass a Pandas dataframe (False) or numpy arrayTrue).  Mainly for compatibility
                    with scikit which requires arrays,
        :return:
            What the decorated function returns.
        """

        def decorator(fun):
            # i suppose i could of done @wraos for self, but this works
            def wrapper(*args, **kwargs):
                objecty = args[0]
                if not objecty.trained:
                    raise_PasoError("Train must be called before predict")
                objecty.measure = _dict_value2(
                    kwargs, objecty.description_kwargs, "measure", ""
                )
                if len(args) != narg:
                    raise_PasoError(
                        "PredictWrap:Must be {} arguments. was:{} ".format(narg, args)
                    )
                if len(args) >= 2:
                    Xarg = args[1]
                if objecty.trained == False:
                    raise PasoError(
                        "Predict:Must call train before predict.", args, kwargs
                    )
                # must be dataFrame
                if is_DataFrame(Xarg):
                    pass
                else:
                    raise raise_PasoError(
                        "PredictWrap:Xarg must be if type DataFrame. Was type:{}".format(
                            type(Xarg)
                        )
                    )
                # cached . dont'caclulate again

                _Check_No_NA_Values(Xarg)

                if array:
                    result = fun(objecty, Xarg.to_numpy(), **kwargs)
                else:
                    result = fun(objecty, Xarg, **kwargs)

                objecty.predicted = True
                return result

            return wrapper

        return decorator


### log class
class Log(object):
    """
        Fetch existing ``paso`` log instance.

        Parameters:
            None

        Return:
            self
    """

    log_ids = {}
    log_names = {}
    count = 0
    additional_log_files = {}

    def __init__(self, verbose=True):
        self.sink_stdout = None
        self.verbose = verbose

    def log(self, log_name="paso", log_file=""):
        """
        Retrieves log instance by <log_name> or creates a new log instance
        by <log_name>.


        Parameters:
            log_name (string) Top level title used un logger
                default:  'paso'

       Class Attributes:
            log_ids
                init: {}
            log_names
                init: {}
            count = 0
                init: 0
            additional_log_files
                init: {}

        filepath :base
            default: ../parameters/base.yaml

        Returns:
            ``loguru`` object instance

        Example:
            >>> from paso.base import Paso
            >>> session =  Paso().startup()
            `` paso 4.6.2019 12:16:19 INFO Log started
                paso 4.6.2019 12:16:19 INFO ========================================
                paso 4.6.2019 12:16:19 INFO Read in parameter file: ../parameters/base.yaml``
            >>> x =1
            >>> logger.info("x:{}".format(x))

        """

        # same or new log
        if log_name not in Log.log_names:
            logger.remove()  # remove previous handlers
            Log.count += 1
            if log_file == "":
                self.sink_stdout = logger.add(
                    sys.stdout,
                    format=log_name + " {time:D.M.YYYY HH:mm:ss} {level} {message}",
                )
            else:
                self.sink_stdout = logger.add(
                    log_file,
                    format=log_name + " {time:D.M.YYYY HH:mm:ss} {level} {message}",
                )

            Log.log_names[log_name] = "On"
            Log.log_ids[log_name] = self.sink_stdout
            if self.verbose:
                logger.info("Log started")

        return self

    def file(self, log_name="paso", log_file=""):
        if log_file == "":
            log_file = "default.log"
        Log.additional_log_files[log_file] = logger.add(
            log_file, format=log_name + " {time:D.M.YYYY HH:mm:ss} {level} {message}"
        )
        if self.verbose:
            logger.info("Logging also to file:{}".format(log_file))
        return self


### Param class
class Param(object):
    """
    Read in from file(s) the parameters for this service.

        base.yml
        experiment-1 (optional)
        ....         (optional)

        Class Attributes:

            Param.filepath
                init: None

            Param.parameters_D  (dict) Parameter dictionary resulting from reading in ``filepath``.
                init: None


            Setting Class attribute to ``None`` means it initialized by whatever method set it.

    """

    gfilepath = None
    parameters_D = None

    def __init__(self, filepath="", verbose=True):
        """
        Read-in a parameter file on ``filepath`.

        Parameters:
            None

        Returns:
            self (definition of __init__ behavior).

        Note:
            Currently bootstrap to <name>.yaml or from attribute ``experiment_environment``
            from default ``../parameters/base.yaml``  thus any <na00me>.yaml can be used
            without change to current code. Only new code need be added to
            support <nth name>.yaml.

            Notice instance is different on call to class init but resulting
            parameter dictionary is always the same as the file specified by
            parameter  ``experiment_environment``. This means class parameters
            can be called from anywhere to give the same parameters and values. so long a Parm file path

            It also means if dafault.yaml or underlying file specified by
            `experiment_environment`` is changed, parameters class instance is set
            again with a resulting new parameters dictionary.

        Example:

            >>> p = Param.parameter_D
            >>> p['a key']

        """
        if filepath == "":
            filepath = "../parameters/base.yaml"

        Param.gfilepath = filepath
        Param.parameters_D = self._read_parameters(filepath)
        self.parameters_D = Param.parameters_D

    def _read_parameters(self, filepath):
        if os.path.exists(filepath):
            with open(filepath) as f:
                config = yaml.load(f)
                return AttrDict(config)
        else:
            raise_PasoError(
                "read_parameters: The file does not exist:{}".format(filepath)
            )


### Paso class
class Paso(object):
    """
    Creates
        1. Log: default name paso
        2. parameter file : default: '../parameters/base.yaml'
        3. list of pasoes invoked.

        Parameters:
            log_name (str) 'paso'
            verbose: (boolean) True

        Class Attributes:
            pipeLine (list)
                init: []

        Class Instance Attributes::
            self.parameters = None
            self.log_name = log_name
            self.log_file = log_file
            validate_bool_kwarg(verbose, "verbose")
            self.verbose = verbose

    """

    pipeLine = []

    def __init__(
        self, verbose=True, log_name="paso", log_file="", parameters_filepath=""
    ):

        self.log = None

        if parameters_filepath == "":
            self.parameters_filepath = "../parameters/base.yaml"
        else:
            self.parameters_filepath = parameters_filepath

        self.parameters = None
        self.log_name = log_name
        self.log_file = log_file
        validate_bool_kwarg(verbose, "verbose")
        self.verbose = verbose

    def __enter__(self):
        return self.startup()

    def __exit__(self, *args, **kwargs):
        return self.shutdown()

    #   def __str__(self):
    #        return "{}".format(self)

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, value):
        self._log = value

    @property
    def log_name(self):
        return self._log_name

    @log_name.setter
    def log_name(self, value):
        self._log_name = value

    @property
    def log_file(self):
        return self._log_file

    @log_file.setter
    def log_file(self, value):
        self._log_file = value

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def parameters_filepath(self):
        return self._parameters_filepath

    @parameters_filepath.setter
    def parameters_filepath(self, value):
        self._parameters_filepath = value

    def startup(self):
        Log(verbose=self.verbose).log(log_name=self._log_name, log_file=self._log_file)
        if self.verbose:
            logger.info("========================================")
        Param(filepath=self.parameters_filepath, verbose=self.verbose)
        if self.verbose:
            logger.info("Read in parameter file: {}".format(self.parameters_filepath))
        #        self.flush()
        Paso.pipeLine.append(["Startup Paso", "square", " "])
        return self

    def shutdown(self):
        Paso.pipeLine.append(["Shutdown Paso", "square", " "])
        if "paso" in Log.log_names:
            del Log.log_names["paso"]
        if "paso" in Log.log_ids:
            logger.remove(Log.log_ids["paso"])
            del Log.log_ids["paso"]

        logger.add(
            sys.stdout, format=" default {time:D.M.YYYY HH:mm:ss} {level} {message}"
        )
        return self

    def display_DAG(self):
        """
        Displays pipeline structure in a jupyter notebook.

        Param:
            pipe (list of tuple): (node_name,shape,edge_label

            Returns:
                graph (pydot.Dot): objecty representing upstream pipeline paso(es).
        """
        graph = self._pasoLine_DAG(__PiPeLiNeS__)
        plt = Image(graph.create_png())
        display(plt)
        return self

    def DAG_as_png(self, filepath):
        """
        Saves pipeline DAG to filepath as png file.

        Parameters:
            pipe (list of tuple): (node_name,shape,edge_label

            filepath (str): filepath to which the png with pipeline visualization should be persisted

        Returns:
            graph (pydot.Dot): objecty representing upstream pipeline paso(es).
        """
        graph = self._pasoLine_DAG(__PiPeLiNeS__)
        graph.write(filepath, format="png")
        return self

    def _pasoLine_DAG(self, pipe):
        """
        DAG of the pasoline.

        Parameters:
            pipe (list of tuple): (node_name,shape,edge_label

        Returns:
            graph (pydot.Dot): objecty representing upstream pipeline paso(es).

        """
        graph = pydot.Dot()
        node_prev = None
        label_prev = "NA"
        for p in pipe:
            if p[1] == "":
                node = pydot.Node(p[0])
            else:
                node = pydot.Node(p[0], shape=p[1])

            graph.add_node(node)

            if node != node_prev and node_prev != None:
                graph.add_edge(pydot.Edge(node_prev, node, label=label_prev))
            label_prev = p[2]
            node_prev = node
        return graph


### pasoBase class
class pasoBase(ABC):
    """
    pasoBaseClass is a base class for creating a **paso**, which is
    a building block for a data analytics pipeline or any pipeline of
    computation that can be specified as a directed acyclic graph(DAG).

    **paso** has these key objectyives:

    Simplicity:
        for now this means objecty-oriented with as few as attributes as possible
        and a standard set of a few method calls.  You have a choice, though the
        common parameter ``inplace``, to transform or not change input arguments.
        the latter promotes a functional model that lends itself to high parallelism.

    Performance:
        hides performance increases due to refactoring
        because of cpython, dask, spark, GPUs, multi-threading, quantum computing, etc.

    There are two major types of **paso** base classes:

    **Class pasoModelBase Methods**:

        Models calculate by gradient descent method to minimize some cost function,
        (also known as a loss or error function)) during ``train``,
        the weights (local deep learning vernacular) ,
        co-efficients, vector, maxtrix, or tensor. We shall
        refer to these fitted parameters on input `` x`` as
        ``w`` such that ``f(x) = x dot w``, where ``f(x)``
        is the **target** or goal of the optimization.

            - ``train`` calculates ``w`` the weights of the model such that ``f(y) = y dot w`` for class ``pasoModelBase``

            - ``predict`` calculates ``f(y)`` by the model weights ``w`` such that ``f(y) = y dot w``.

            - ``save`` puts the model, usally the weights, ``w``, to permanent storage. ``filepath`` must be given to access a specific named file.

            - ``load`` retrieves the model, usally the weights, ``w``,from permanent storage. Optionally ``filepath`` maybe given to access a specific named file.

            - ``reset`` Reset **paso** objecty to default state.

    **Class pasoFuctionBase Methods**:

        A function, called a  tranformer,  which is pythonic/sklearn slang for a
        function ``f`` operating on the input ``x``. Unlike the pasoModelBase
        class. the pasoFuctionBase base class
        does not need to derive ``f`` though training, but rather ``f`` is a
        given, of some specifiable form,  that changes input ``x`` to ``f(x)`` .

        - ``transform`` calculates ``f(x)`` with input parameter ``x``  for class ``pasoFunctionBase`` .

    **Class Methods common to both pasoFunctionBase and pasoModelBase**:

        - ``write``  puts ``f(x)`` to permanent storage for ``pasoFunctionBase`` .

        - ``read``  gets ``f(x)`` from permanent storage for ``pasoFunctionBase`` .

        - ``reset`` Reset **paso** object to default state.

    +-----------------+------------------+---------------+
    | flag            | pasoFunctionBase | pasoModelBase |
    +=================+==================+===============+
    | **method**      |      write       |      save     |
    +-----------------+------------------+---------------+
    | transformed     |       T          |     None      |
    +-----------------+------------------+---------------+
    | checkpoint      |       T          |     T         |
    +-----------------+------------------+---------------+
    | trained         |       None       |     T         |
    +-----------------+------------------+---------------+
    | predicted       |       None       |     T         |
    +-----------------+------------------+---------------+
    | **method**      |        read      |      load     |
    +-----------------+------------------+---------------+
    | checkpoint      |        T         |      T        |
    +-----------------+------------------+---------------+
    |   file_name     | read Transform   | read Train    |
    |                   checkpoint filename              |
    +-----------------+------------------+---------------+
    | model_checkpoint|       None       |      T        |
    +-----------------+------------------+---------------+
    | model_fileexists | read Transform   | read Predict |
    |                   checkpoint_model_filename        |
    +-----------------+------------------+---------------+
    """

    # todo fix read/write chechpoint in train/predict/transform
    # move all kw into train/predict/transform esp. inplace,verbose
    def __init__(self, **kwargs):
        # function and model
        self.checkpoint = False  # for transform or train
        self.checkpoint_file_name = ""  # for transform or train


### pasoFuction class
class pasoFunction(pasoBase):
    """
    For transform functions ``f(x)`` only.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # transform of x tp f(x) has been accomplished on this instance
        self.transformed = False

    def reset(self):
        """
        Reset **paso** object to default state.

        Parameters:
                None

        Returns:
            self
        """

        if os.path.exists(self.checkpoint_file_name):
            os.remove(self.checkpoint_file_name)

        self.transformed = False
        self.checkpoint = False  # for transform or train
        self.checkpoint_file_name = ""  # for transform or train

        return self

    def write(self, X):
        """
        Writes X to local checkpoint_file_name to directory given in ˜˜pasoparameters/checkpoint.yaml.

        Parameters:
            X: (dataframe)

        Returns:
            self

        Raises:
            PasoError(\"must have non-blank checkpoint_file_name.\")

            PasoError(\"Must transform before checkpoint operation.\"  )

        """
        if self.transformed:
            if self.checkpoint_file_name != "":
                self.checkpoint = True
                X.to_parquet(self.checkpoint_file_name)
                return self
            else:
                raise_PasoError("must have non-blank checkpoint_file_name.")
        else:
            raise_PasoError("Must transform before checkpoint operation.")

    def read(self):
        """
        Reads the transform X from the ``checkpoint_file_name`` from
        directory given in ˜˜pasoparameters/checkpoint.yaml
        Uses ``filepath`` previously used if passed parameter ``filepath``
        is blank.

        Parameters:
            self
        Returns
            What is in self.checkpoint_file_name
            or False if self.checkpoint_file_name does not exist

        Raises:
            PasoError(\"Must write f_x before read.\"  )
        """
        if self.checkpoint_file_name == "":
            raise_PasoError("must have non-blank checkpoint_file_name.")

        if os.path.exists(self.checkpoint_file_name):
            result = pd.read_parquet(self.save_file_name)
            return result
        else:
            return False

    def transform(self, X, **kwargs):
        """
        Performs transformation f(x) of data, x.
        Parameters:
            args:
                positional arguments (can be anything)
            kwargs:
                keyword arguments (can be anything)
        Returns:
            f(X:) (dataframe)
        """
        raise NotImplementedError


### PasoModel class
class pasoModel(pasoBase):
    """
    For model only.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # model specific
        self.trained = False  # train has been accomplied on this instance
        self.predicted = False  # predict has been accomplied on this instance
        self.checkpoint_model_file_name = ""

    def reset(self):
        """
        Reset **paso** object to default state.
        Parameters:
            None
        Returns:
            self
        """

        if os.path.exists(self.checkpoint_file_name):
            os.remove(self.checkpoint_file_name)

        self.trained = False  # train has been accomplied on this instance
        self.checkpoint = False  # for transform or train
        self.checkpoint_file_name = ""  # for transform or train

        if os.path.exists(self.checkpoint_model_file_name):
            os.remove(self.checkpoint_model_file_name)

        self.predicted = False  # predict has been accomplied on this instance
        self.checkpoint_model = False
        self.checkpoint_model_file_name = ""

        return self

    def write(self, X):
        """
        Writes X to local checkpoint_file_name to directory given in ˜˜pasoparameters/checkpoint.yaml.
        Parameters:
            X: (dataframe)
        Returns:
            self
        Raises:
            PasoError(\"must have non-blank checkpoint_file_name.\")

            PasoError(\"Must transform before checkpoint operation.\"  )
        """
        if self.trained:
            if self.checkpoint_file_name != "":
                self.checkpoint = True
                X.to_parquet(self.checkpoint_file_name)
                return self
            else:
                raise_PasoError("must have non-blank checkpoint_file_name.")
        else:
            raise_PasoError("Must transform before checkpoint operation.")

    def read(self):
        """
        Reads the self.checkpoint_file_name X if non-blank and exists
        from the ``checkpoint_file_name`` from
        directory given in ˜˜pasoparameters/checkpoint.yaml
        Otherise returns False

        Parameters:
            self

        Returns:
            What is in self.checkpoint_file_name
            or False if self.checkpoint_file_name does not exist or is blank str.

        Raises:
            PasoError(\"must have non-blank checkpoint_file_name."\"  )
        """
        if self.checkpoint_file_name == "":
            raise_PasoError("read:: must have non-blank checkpoint_file_name.")

        if os.path.exists(self.checkpoint_file_name):
            result = pd.read_parquet(self.save_file_name)
            return result
        else:
            return False

    def load(self):
        """
        Reads the self.checkpoint_model_file_name X if non-blank and exists
        from the ``checkpoint_model_file_name`` from
        directory given in ˜˜pasoparameters/checkpoint.yaml
        Othewise returns False

        Model specific.
            - checkpoint_model
            - checkpoint_model_file_name
        """
        raise NotImplementedError

    def save(self, X):
        """
        Writes X to local checkpoint_file_name to directory given in `paso/parameters/checkpoint.yaml```.
        
        Model specific.
            - checkpoint_model
            - checkpoint_model_file_name
        Parameters:
            X: (dataframe)
        Returns:
            self
        Raises:
            PasoError(\"must have non-blank checkpoint_file_name.\")

            PasoError(\"Must transform before checkpoint operation.\"  )
        """
        raise NotImplementedError

    def train(self, X, **kwargs):
        """
        Calulation using ``x`` of trainable parameters. for ``f(x)``.
        All model determines parameters based on input data, ``x`` .
        Notes:
            The trained parameters can be cached beyond the train/predict cycle.
        Parameters:
            X: (dataframe)
            args:
                positional arguments (can be anything)
            kwargs:
                keyword arguments (can be anything)
        Returns:
            self (pasoModel instance)
        Raises:
             NotImplementedError
        """
        raise NotImplementedError

    def predict(self, X, **kwargs):
        """
        Performs prediction f(y) of data, y, using model trained parameters
        that define the transform function **f**.

        Notes:
            All data prediction including prediction with deep learning/machine learning models
            will be performed by this method.
            If trainng of parameters is required by ``predict`` . there will be a check that
            train has been performed.  The trained model can be persisted or loaded from long-term
            memory using **save/load**,

            The reult of the last predict transform, **f(y)** can be written or read from long-term
            memory using **write/read**. You might do this if prediction takes some amount of time,
            and you don't need to recalulate.

        Parameters:
            X: (dataframe)
            args:
                positional arguments (can be anything)
            kwargs:
                keyword arguments (can be anything)

        Returns:
            f(y): (dataframe)
        Raises:
             NotImplementedError
        """
        raise NotImplementedError


### toDataFrame class
# makes no sense to save,load or persist toDataframe
class toDataFrame(pasoFunction):
    """
    A paso to transform a  list, tuple, numpy 1-D or 2-D array, or Series
    into a pandas DataFrame.
    Usually used to transform a paso numpy array into a Dataframe.

    Parameters:
        None

    Example:

        >>> toDataFrame(df)

    """

    def __init__(self, verbose=True):
        super().__init__()

    def transform(self, Xarg, labels=[], **kwargs):
        """
        Transform a list, tuple, numpy 1-D or 2-D array, or pandas Series  into a  DataFrame.

        Parameters:
            Xarg: (list, tuple, numpy 1-D or 2-D array, pandas Serieandas dataFrame, pandas dataFrame)

            labels: (single feature name strings or list of feature name strings):
                The column labels name to  be used for new DataFrame.
                If number of column names given is less than number of column names needed,
                then they will generared as Column_0...Column_n, where n is the number of missing column names.

            inplace: boolean, (CURRENTLY IGNORED)
                Xarg is type DataFrame then return Xarg , inplace = True
                Xarg is NOT type DataFrame then return a created Dataframe made from Xarg , inplace = False

                    False (, replace 1st argument with resulting dataframe
                    True:  (boolean) ALWAYS False

        Note:
            A best practice is to make your dataset of type ``DataFrame`` at the start of your pipeline
            and keep the orginal DataFrame thoughout the pipeline of your experimental run to maximize
            speed of completion and minimize memory usage, THIS IS NOT THREAD SAFE.

            Almost all **paso** objects call ``toDataFrame(argument)`` ,which if argument
            is of type ``DataFrame``is very about 500x faster, or about 2 ns  for ``inplace=False``
            for single thread for a 1,000,000x8 DataFrame.

            If input argument is of type DataFrame,
            the return will be passed DataFrame as if `inplace=True```and ignores ``labels``

            If other than of type ``DataFrame`` then  `inplace=False``, and  `inplace`` is ignored
            and only remains for backwaeds compatability.

        Returns: (pandas DataFrame)

        Raises:
            1. ValueError will result of unknown argument type.
            2. ValueError will result if labels is not a string or list of strings.
       """

        if is_DataFrame(Xarg):
            self.transformed = True
            self.f_x = Xarg
            return Xarg  # new labels not set on inplace DataFrame, inplace ignored
        # for now inplace ignored
        X = Xarg.copy()
        if len(X) == 0:
            raise_PasoError(
                "toDataFrame:transform:X: is of length O: {} ".format(str(type(X)))
            )

        self.labels = labels

        if is_Series(X):
            pass
        elif type(X) == np.ndarray:
            if X.ndim != 2:
                raise_PasoError(
                    "toDataFrame:transform:np.array X: wrong imension. must be 2: {} ".format(
                        str(X.ndim)
                    )
                )
            if labels == []:
                for i in range(X.shape[1]):
                    self.labels.append("c_" + str(i))
        elif type(X) == tuple or type(X) == list:
            # if someone passes a nested list or tuple of labels, well i am nt going to check
            self.labels = ["c_0"]
        else:
            raise raise_PasoError(
                "S_toDataFrame:transform:X: is of wrong type: {} ".format(str(type(X)))
            )

        X = pd.DataFrame(X)
        self.transformed = True
        return _new_feature_names(X, self.labels)


@jit
def _time_required(func):
    timerr = timeit.default_timer
    x = timerr()
    for i in range(10):
        func()
    y = timerr()
    return y - x


@jit
def _time_required_dask(func):
    timerr = timeit.default_timer
    x = timerr()
    for i in range(10):
        func().compute()
    y = timerr()
    return y - x


def dask_pandas_startup_ratio(magnitude=1):
    """
    dask_cost calulates the ratio of dask dataframe - pandas
    dataframe in wall clock time. On a single CPU the ratio is 1.0.
    On a multiprocessor (usually 2 threads per CPU) the dask-pandas time
    cost ratio grows larger (favorable to Dask)
    the more elements per benchmark dataframe.

    Parameters:
        Nome

    Returns:
        pandas dataframe

    Example:
        >>> (On 12-CPU MacPro)
        >>> dask_pandas_ratio()
    """

    from sklearn.datasets import load_boston
    import dask.dataframe as pdd

    boston = load_boston()
    City = pd.DataFrame(boston.data, columns=boston.feature_names)

    c = []
    m = []
    for power in tqdm(range(magnitude)):
        scale = 10 ** power
        bc = pd.concat([City for i in range(int(1.5 * scale))], axis=0)
        if self.verbose:
            logger.info((type(bc), bc.shape, bc.shape[0] * bc.shape[1]))
        N = mp.cpu_count()  # theads on this machine
        bcd = pdd.from_pandas(bc, npartitions=N)

        t1 = _time_required(bc.count)
        t2 = _time_required(bcd.count)
        c.append(
            (
                "count",
                round(math.log10(bc.shape[0] * bc.shape[1]), 0),
                t1,
                t2,
                (round(t1 / t2, 2)),
            )
        )

        t1 = _time_required(bc.sum)
        t2 = _time_required(bcd.sum)
        c.append(
            (
                "sum",
                round(math.log10(bc.shape[0] * bc.shape[1]), 0),
                t1,
                t2,
                (round(t1 / t2, 2)),
            )
        )

        t1 = _time_required(bc.mean)
        t2 = _time_required(bcd.mean)
        c.append(
            (
                "mean",
                round(math.log10(bc.shape[0] * bc.shape[1]), 0),
                t1,
                t2,
                (round(t1 / t2, 2)),
            )
        )

        t1 = _time_required(bc.isnull)
        t2 = _time_required(bcd.isnull)
        c.append(
            (
                "isnull",
                round(math.log10(bc.shape[0] * bc.shape[1]), 0),
                t1,
                t2,
                (round(t1 / t2, 2)),
            )
        )

    df = pd.DataFrame(
        c, columns=["f()", "log10(N)", "t-pd(s)", "t-dask(s)", "t-pd/t-dask"]
    )
    ax = sns.lineplot(x="log10(N)", y="t-pd/t-dask", hue="f()", data=df)
    ax.set_yscale("log")
    ax.set_xlabel("Number elements(log10)")
    ax.set_ylabel("Ratio dask/pandas")
    plt.plot([4, 8], [1, 1], c="black", linewidth=3)
    return df


def dask_pandas_ratio(magnitude=4):
    """
    dask_cost calulates the ratio of dask dataframe - pandas
    dataframe in wall clock time. On a single CPU the ratio is 1.0.
    On a multiprocessor (usually 2 threads per CPU) the dask-pandas time
    cost ratio grows larger (favorable to Dask)
    the more elements per benchmark dataframe.

    Parameters:
        magnitude:  (int), default 4

    Returns:
        pandas dataframe

    Example:
        >>> (On 12-CPU MacPro)
        >>> dask_pandas_ratio()
    """

    from sklearn.datasets import load_boston
    import dask.dataframe as pdd

    boston = load_boston()
    City = pd.DataFrame(boston.data, columns=boston.feature_names)

    c = []
    m = []
    for power in tqdm(range(magnitude)):
        scale = 10 ** power
        bc = pd.concat([City for i in range(int(1.5 * scale))], axis=0)
        N = mp.cpu_count()  # theads on this machine
        bcd = pdd.from_pandas(bc, npartitions=N)

        t1 = _time_required(bc.count)
        t2 = _time_required_dask(bcd.count)
        c.append(
            (
                "count",
                round(math.log10(bc.shape[0] * bc.shape[1]), 0),
                t1,
                t2,
                (round(t1 / t2, 2)),
            )
        )

        t1 = _time_required(bc.sum)
        t2 = _time_required_dask(bcd.sum)
        c.append(
            (
                "sum",
                round(math.log10(bc.shape[0] * bc.shape[1]), 0),
                t1,
                t2,
                (round(t1 / t2, 2)),
            )
        )

        t1 = _time_required(bc.mean)
        t2 = _time_required_dask(bcd.mean)
        c.append(
            (
                "mean",
                round(math.log10(bc.shape[0] * bc.shape[1]), 0),
                t1,
                t2,
                (round(t1 / t2, 2)),
            )
        )

        t1 = _time_required(bc.isnull)
        t2 = _time_required_dask(bcd.isnull)
        c.append(
            (
                "isnull",
                round(math.log10(bc.shape[0] * bc.shape[1]), 0),
                t1,
                t2,
                (round(t1 / t2, 2)),
            )
        )

    df = pd.DataFrame(
        c, columns=["f()", "log10(N)", "t-pd(s)", "t-dask(s)", "t-pd/t-dask"]
    )
    ax = sns.lineplot(x="log10(N)", y="t-pd/t-dask", hue="f()", data=df)
    ax.set_yscale("log")
    ax.set_xlabel("Number elements(log10)")
    ax.set_ylabel("Ratio dask/pandas")
    plt.plot([4, 8], [1, 1], c="black", linewidth=3)
    return df
