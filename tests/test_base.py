import pytest
#
from paso.base import pasoFunction, pasoModel,pasoError
from paso.base import Paso, Param, Log
from paso.base import dask_pandas_ratio
import loguru as logger
#from attrdict import AttrDict


def test_paso_count_init():
    assert Log.count == 0

def test_paso_startup():
    Paso().startup()
    assert Log.count == 1

def test_paso_namePaso():
    Paso().startup()
    assert Log.log_names == {'paso': 'On'}

def test_paso_claas_Paso():
    assert str(type(    Paso().startup())) == "<class 'paso.base.Paso'>"

def test_paso_logger_return():
    Paso().startup()
    assert Param().parameters_D == {'project': 'Common Ground Soltions/paso', 'name': 'lesson.1'
        , 'cpu_n': 12, 'threads_n': 24, 'gpu_n': 2, 'cv_n': 5, 'shuffle': 1, 'layer-1': {'layer-2': 'value'}}

def test_parm_default():
    Paso().startup()
    assert Param().parameters_D['project'] == 'Common Ground Soltions/paso'

def test_parm_file_error():
    with pytest.raises(ValueError):
        Paso('fred' ).startup()

def test_parm_startup_arg_error():
    with pytest.raises(TypeError):
        Paso().startup('fred')