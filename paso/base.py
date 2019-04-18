#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

from abc import ABC

# todo: insert dask datatypes all over
import logging
import pydot_ng as pydot
from IPython.display import Image, display
import sys, os
import yaml
from attrdict import AttrDict
import numpy as np
from numba import jit
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import seaborn as sns
import copy
import pandas as pd
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndexClass, ABCSeries
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.common import (
    is_bool,
    is_categorical_dtype,
    is_datetime64_dtype,
    is_datetimelike,
    is_dict_like,
    is_extension_array_dtype,
    is_extension_type,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_scalar,
    is_string_like,
    is_timedelta64_dtype,
)
import dask.dataframe as pdd
import multiprocessing as mp
import timeit, random, math

#
import warnings

warnings.filterwarnings("ignore")
# paso imports
# from paso.base import pasoFunction, pasoLine,get_paso_log


def _Check_No_NA_F_Values(df, feature):

    if not df[feature].isna().any():
        return True
    else:
        raise pasoError("Passed dataset, 1st arg, contained NA")
        return False


def _Check_No_NA_Values(df):
    #    import pdb; pdb.set_trace() # debugging starts here
    for feature in df.columns:
        if _Check_No_NA_F_Values(df, feature):
            pass
        else:
            return False
    return True


class pasoError(Exception):
    pass


class pasoDecorators:
    def TransformWrap(fun):
        # i suppose i could of done @wraos for self, but this works
        def wrapper(*args, **kwargs):
            object = args[0]
            if len(args) < 2:
                raise pasoError(
                    "TransformWrap:Must be at keast two arguments: ", args, kwargs
                )
            else:
                Xarg = args[1]
            object.inplace = False
            object.drop = False
            kwa = "inplace"
            if kwa in kwargs:
                object.inplace = kwargs[kwa]
                validate_bool_kwarg(object.inplace, kwa)
            kwa = "drop"
            if "drop" in kwargs:
                object.drop = kwargs[kwa]
                validate_bool_kwarg(object.drop, kwa)
            # must be dataFrame
            if is_DataFrame(Xarg):
                pass
            else:
                raise pasoError(
                    "TransformWrap:Xarg must be if type DataFrame. Was type:{}",
                    format(type(Xarg)),
                )
            # cached . dont'caclulate again
            if object.cache and object.transformed:
                return object.f_x
            else:
                pass
            if object.inplace:
                X = Xarg
            else:
                X = Xarg.copy()
            _Check_No_NA_Values(X)
            # pre
            fun(object, X, **kwargs)
            # post
            if object.cache:
                object.f_x = X
            object.transformed = True
            return X

        return wrapper


class pasoBase(ABC):
    """
    pasoBaseClass is a base class for creating a **paso**, which is
    a building block for a data analytics pipeline or any pipeline of
    computation that can be specified as a directed acyclic graph(DAG).

    **paso** has these key objectives:

    Simplicity:
        for now this means object-oriented with as few as attributes as possible
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

        - ``cacheOn`` local cache (in-memory storage) of ``f(x)`` is turned ON.

        - ``cacheOff`` (default) local cache (in-memory storage) of ``f(x)`` is turned OFF.


    +-----------------+------------------+---------------+
    | flag            | pasoFunctionBase | pasoModelBase |
    +=================+==================+===============+
    | **method**      |      write       |      save     |
    +-----------------+------------------+---------------+
    |train_required   |       None       |     T         |
    +-----------------+------------------+---------------+
    | trained         |       None       |     T         |
    +-----------------+------------------+---------------+
    | transformed     |       T          |     None      |
    +-----------------+------------------+---------------+
    | model_persisted |       F->T       |     F->T      |
    +-----------------+------------------+---------------+
    | model_cacheOn   |       cache->T   |      cache->T |
    +-----------------+------------------+---------------+
    | cacheOff        |       cache->F   |      cache->F |
    +-----------------+------------------+---------------+
    | **method**      |        read      |      load     |
    +-----------------+------------------+---------------+
    | persisted       |        T         |      T        |
    +-----------------+------------------+---------------+
    | cacheOn         |        cache>T   |      cache->T |
    +-----------------+------------------+---------------+
    | cacheOff        |        F         |      F        |
    +-----------------+------------------+---------------+
    """

    def __init__(self):
        # common
        self._cache = False
        self.persisted = False  # has not been persisted to file yet
        self.save_file_name = ""
        # function
        self.f_x = None
        self.transformed = (
            False
        )  # transform of x tp f(x) has been accomplied on this instance
        # model
        self.model = None
        self.trained = False  # train has been accomplied on this instance
        self.predicted = False  # predict has been accomplied on this instance
        self._model_cache = False
        self.model_persisted = False
        self.save_model_file_name = ""

    @property
    def model_cache(self):
        """
        Attribute:s
            cache: (boolean) False
        """
        return self._model_cache

    @model_cache.setter
    def model_cache(self, value):
        self._model_cache = value

    def model_cacheOn(self):
        """
        Turn _model_caching on.

        Parameters:
             None

        Returns:
            Self
        """
        self.model_cache = True
        return self

    def model_cacheOff(self):
        """
        Turn _model_caching off.

        Parameters:
            None

        Returns:
            Self
        """
        self.model_cache = False
        return self

    @property
    def cache(self):
        """
        Attribute:s
            cache: boolean (False)
        """
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    def cacheOn(self):
        """
        Turn caching on.

        Parameters:
             None

        Returns:
            Self
        """
        self.cache = True
        return self

    def cacheOff(self):
        """
        Turn caching off.

        Parameters:
            None

        Returns:
            Self
        """
        self.cache = False
        return self


class pasoModel(pasoBase):
    """
    For model only.
    """

    def __init__(self):
        super().__init__()

        # model
        self.transformed = None
        self.model = None
        self.trained = False  # train has been accomplied on this instance

    def reset(self):
        """
        Reset **paso** object to default state.

        Parameters
            None

        Returns:
            self
        """
        # common
        self._cache = False
        # function
        self.f_x = None  # lower pointer count

        self.transformed = None  # not transformed yet
        # model
        self._model_cache = False
        self.model_persisted = False
        self.trained = False  # not trained yet
        self.predicted = False
        self.model = None

        if self.persisted and (self.save_file_name != ""):
            if os.path.exists(self.save_file_name):
                os.remove(self.save_file_name)
                self.persisted = False
                self.save_file_name = ""
            else:
                raise NameError(
                    "reset: The f(x) file does not exist:{}".format(self.save_file_name)
                )

        if self.model_persisted and (self.save_model_file_name != ""):
            if os.path.exists(self.save_model_file_name):
                os.remove(self.save_model_file_name)
                self.model_persisted = False
                self.save_model_file_name = ""
            else:
                raise NameError(
                    "reset: The model file does not exist:{}".format(
                        self.save_model_file_name
                    )
                )

        return self

    def load(self, filepath):
        """
        Loads the trainable parameters of the mode

        Parameters:
            filepath: (str)
                filepath from which the model should be read.
                If ``filepath``is blank then last valid ``filepath``i will be used.

        Returns:
            f(X:) (model object)

        Raises:
            pasoError(\"Must write model before load.\"  )
        """
        if filepath != "":
            self.save_model_file_name = filepath
        # file does not exist
        if not os.path.exists(self.save_model_file_name):
            raise NameError(
                "load model:The file does not exist:{}".format(
                    self.save_model_file_name
                )
            )
        if self.model_persisted:
            return self.model.load(self.save_model_file_name)
        else:
            raise pasoError("Must save f_x before load.")

    def save(self, filepath):
        """
        Saves the trainable weights of the model

        Parameters:
            filepath: (str)
                filepath where the predict parameters should be persisted

        Raises:
                raise pasoError("Model:save must have non-blank filename.")
        else:
            raise pasoError("Model: Must train and predict before save operation."  )
        """
        if self.trained:
            if filepath != "":
                self.model_persisted = True
                self.save_model_file_name = filepath
                if not os.path.exists(self.save_model_file_name):
                    raise NameError(
                        "save model:The file does not exist:{}".format(
                            self.save_model_file_name
                        )
                    )
                self.model.save(self.save_model_file_name)
                return self
            else:
                raise pasoError("Model:save must have non-blank filename.")
        else:
            raise pasoError("Model: Must train and predict before save operation.")

    def write(self, filepath=""):
        """
        Writes ``f(y)`` to local fs if ``filepath`` given.

        Parameters:
            filepath: (str), default: '
                filepath where the predict f(y) should be written.
                An error will result if not passed a valid path string.

        Returns:
            f(X:) (dataframe)

        Raises:
            pasoError(\"Model:write must have non-blank filename.\")

            pasoError(\"Must write f_x before read.\"  )

        """
        if self.trained and self.predicted:
            if filepath != "":
                self.persisted = True
                self.save_file_name = filepath
                if not os.path.exists(self.save_file_name):
                    raise NameError(
                        "write f(x)l:The file does not exist:{}".format(
                            self.save_file_name
                        )
                    )
                self.f_x.to_parquet(filepath)
                return True
            else:
                raise pasoError("Model: must have non-blank filename.")
        else:
            raise pasoError("Model: Must train and predict before write operation.")

    def read(self, filepath=""):
        """
        Reads the transform ``f(x)`` from local fs if ``filepath`` given.

        Parameters:
            filepath: (str)
                filepath from which the transformer should be read.
                If ``filepath``is blank then last valid ``filepath``i will be used.

        Returns:
            f(X:) (dataframe)

        Raises:
            pasoError(\"Must write f_x before read.\"  )
        """
        if filepath != "":
            self.save_file_name = filepath
        if not os.path.exists(self.save_file_name):
            raise NameError(
                "read f(x)l:The file does not exist:{}".format(self.save_file_name)
            )
        if self.persisted:
            self.f_x = pd.read_parquet(self.save_file_name)
            return self.f_x
        else:
            raise pasoError("Must write f_x before read.")

    def train(self, X, *args, **kwargs):
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

    def predict(self, Y, *args, **kwargs):
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


class pasoFunction(pasoBase):
    """
    For transform functions ``f(x)`` only.
    """
    def __init__(self):
        super().__init__()
        # function
        self.f_x = None
        # functional
        self.transformed = (
            False
        )  # transform of x tp f(x) has been accomplied on this instance

        # model
        self.model = None
        self.trained = None  # train has been accomplied on this instance
        self.predicted = None  # train has been accomplied on this instance

    def reset(self):
        """
        Reset **paso** object to default state.

        Parameters:
                None

        Returns:
            self
        """
        # common
        self._cache = False
        # function
        self.f_x = None  # lower pointer count
        self.transformed = None  # not transformed yet
        # model
        self.model = None
        self.trained = False  # not trained yet

        if self.persisted and (self.save_file_name != ""):
            if os.path.exists(self.save_file_name):
                os.remove(self.save_file_name)
                self.persisted = False
                self.save_file_name = ""
            else:
                raise NameError(
                    "reset: The f(x) file does not exist:{}".format(self.save_file_name)
                )

        return self

    def write(self, filepath=""):
        """
        Writes ``f(x)`` to local fs if ``filepath`` given.

        Parameters:
            filepath: str, default: ''
                filepath where the transformer parameters should be written
                An error will result if not paseds a valid path string.

        Returns:
            dataframe: outputs f(x)

        Raises:
            pasoError(\"must have non-blank filename.\")

            pasoError(\"Must write f_x before read.\"  )

        """
        if self.transformed:
            if filepath != "":
                self.persisted = True
                self.save_file_name = filepath
                self.f_x.to_parquet(filepath)
                return True
            else:
                raise pasoError("must have non-blank filename.")
        else:
            raise pasoError("Must predict or transform before write operation.")

    def read(self, filepath=""):
        """
        Reads the transform ``f(x)`` from the ``filepath`` parameter given.
        Uses ``filepath`` previously used if passed parameter ``filepath``
        is blank.

        Parameters:
            filepath: str
                filepath from which the transformer should be read.
                If ``filepath``is blank then last valid ``filepath`` will be used.

        Raises:
            pasoError(\"Must write f_x before read.\"  )
        """
        if filepath != "":
            self.save_file_name = filepath
        if self.persisted:
            self.f_x = pd.read_parquet(self.save_file_name)
            return self.f_x
        else:
            raise pasoError("Must write f_x before read.")

    def transform(self, *args, **kwargs):
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


########## logger
def _start_paso_log(log_name="paso"):

    """
    This log instance is to be used by ``paso``object instances and functions
    to record/log time and messesage as thet execute in the pipeline.

    Parameters:
        None:

    Returns:
        logging.Logger: logger ïnstance

    """

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(
        fmt="%(name)s.%(asctime)s.%(message)s", datefmt="%Y-%m-%d.%H:%M:%S"
    )
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(console_handler)

    return logger

def get_paso_log(log_name="paso"):
    """
    Fetch existing ``paso`` log instance.

    Parameters:
        None:

    Example:

        >>> Log = get_paso_log()
        >>> Log.info('It lives!!!!')
        >>> paso.2019-01-01::12:33:48.It lives!!!

    Returns:
        logging.Logger: logger instance
    """
    return logging.getLogger(log_name)


def _flush_paso_log(log_name="paso"):
    logger = logging.getLogger(log_name)
    #    logger.FileHandler(log_name, mode='w')
    return logger

def _read_paso_parameters(filepath):
    if os.path.exists(filepath):
        with open(filepath) as f:
            config = yaml.load(f)
            return AttrDict(config)
    else:
        raise NameError(
            "read_paso_parameters: The file does not exist:{}".format(filepath)
        )
############# pipeline
# Globals that break every form of concurreny
# todo: eliminate globals

__PiPeLiNeS__ = []


def _set_paso_in_pipe(f_name, f_shape="circle", f_label=""):
    global __PiPeLiNeS__
    __PiPeLiNeS__.append([f_name, f_shape, f_label])  # shape for node, label for edge


class pasoLine:
    """
    paso version of apipline: Iniates:
        1. Log: default name paso
        2. parameter file : default: '../../lessons/parameters/default.yaml'
        3. list of pasoes invoked.
    """

    def __init__(
        self,
        verbose=True,
        log_name="paso",
        parameter_filepath="../lessons/parameters/default.yaml",
    ):
        self._log = None
        self._parameters_filepath = parameter_filepath
        self._parameters_ = None
        self._log_name = log_name
        validate_bool_kwarg(verbose, "verbose")
        self.verbose = verbose

    def __enter__(self):
        return self.startup()

    def __exit__(self, *ATH, **AWAY):
        return self.shutdown()

    def __str__(self):
        return "\n".join([p[0] + " [" + p[2] + "]" for p in __PiPeLiNeS__])

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
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    def startup(self):
        self.log = _start_paso_log(self.log_name)
        self.parameters = _read_paso_parameters(self._parameters_filepath)
        if self.verbose:
            self.log.info(
                "pasoLine: Read parameter file{}".format(self._parameters_filepath)
            )
        self.flush()
        _set_paso_in_pipe("Startup PasoPipeLine", "square", " ")
        return self

    def shutdown(self):
        _set_paso_in_pipe("Shutdown PasoPipeLine", "square", " ")
        print(self)
        #       self.flush()
        logging.shutdown()

        return self

    def flush(self):
        global __PiPeLiNeS__
        __PiPeLiNeS__ = []
        return _flush_paso_log(self._log_name)

    def display_DAG(self):
        """
        Displays pipeline structure in a jupyter notebook.

        Parameters:
            pipe (list of tuple): (node_name,shape,edge_label

            Returns:
                graph (pydot.Dot): object representing upstream pipeline paso(es).
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
            graph (pydot.Dot): object representing upstream pipeline paso(es).
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
            graph (pydot.Dot): object representing upstream pipeline paso(es).

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


def is_Series(X):
    """
    Parameters:
        X (any type)

    Returns:
        True (boolean) if DataFrame Series type.
        False (boolean) otherwise
    """
    return type(X) == pd.core.series.Series


def is_DataFrame(X):
    """
    Parameters:
        X (any type)

    Returns:
        True (boolean) if DataFrame type.
        False (boolean) otherwise
    """
    return type(X) == pd.core.frame.DataFrame


def _new_feature_names(X, labels):
    if labels == []:
        return X
    c = list(X.columns)
    if type(labels) == list:
        c[0 : len(labels)] = labels
    else:
        c[0:1] = [labels]
    X.columns = c
    return X


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

    def transform(self, Xarg, labels=[], inplace=True, **kwargs):
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

        validate_bool_kwarg(inplace, "inplace")

        if is_DataFrame(Xarg):
            self.transformed = True
            self.f_x = Xarg
            return Xarg  # new labels not set on inplace DataFrame, inplace ignored
        # for now inplace ignored
        X = Xarg.copy()
        if len(X) == 0:
            raise ValueError(
                "toDataFrame:transform:X: is of length O: {} ".format(str(type(X)))
            )

        self.labels = labels

        if is_Series(X):
            pass
        elif type(X) == np.ndarray:
            if X.ndim != 2:
                raise TypeError(
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
            raise ValueError(
                "S_toDataFrame:transform:X: is of wrong type: {} ".format(str(type(X)))
            )

        X = pd.DataFrame(X)
        self.f_x = X
        self.transformed = True
        return _new_feature_names(X, self.labels)

# makes no sense to save,load or persist toDataframe

def _time_required(func):
    timerr = timeit.default_timer
    x = timerr()
    for i in range(10):
        func()
    y = timerr()
    return y - x


def dask_pandas_ratio():
    """
    dask_cost calulates the ratio of dask dataframe - pandas
    dataframe in wall clock time. On a single CPU the ratio is 1.0.
    On a multiprocessor (usually 2 threads per CPU) the dask-pandas time
    cost ratio grows larger (favorable to Dask)
    the more elements per benchmark dataframe.

    Parameters:
        None

    Returns:
        None

    Example:
        >>> (On 12-CPU MacPro)
        >>> dask_pandas_ratio()
    """
    import dask.dataframe as pdd
    from sklearn.datasets import load_boston

    boston = load_boston()
    City = pd.DataFrame(boston.data, columns=boston.feature_names)

    c = []
    m = []
    scaleup = 1, 10, 100, 1000, 10000  # ,10000
    for scale in scaleup:
        bcl = [City for i in range(int(1.5 * scale))]
        bc = pd.concat(bcl, axis=0)
        bc.shape, bc.shape[0] * bc.shape[1]
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

    df = pd.DataFrame(c)
    ax = sns.lineplot(x=1, y=4, hue=0, data=df)
    ax.set_yscale("log")
    ax.set_xlabel("Number elements(log10)")
    ax.set_ylabel("Ratio dask/pandas")
    plt.plot([4, 8], [1, 1], c="black", linewidth=3)
    return df
