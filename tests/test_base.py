import pytest
#
from paso.base import Paso, Param, Log
from attrdict import AttrDict


def test_paso_count_init():
    assert Log.count == 0

def test_paso_startup():
    Paso().startup()
    assert Log.count == 1

def test_paso_namePaso():
    Paso().startup()
    assert Log.log_names == {'paso': 'On'}

def test_paso_claas_Paso():
    assert str(type( Paso().startup())) == "<class 'paso.base.Paso'>"

def test_paso_logger_return():
    Paso(parameters_filepath='../parameters/test_base.yaml').startup()
    assert Param.parameters_D ==AttrDict({'project': 'Common Ground Solutions/paso'
                                             , 'name': 'test', 'description': None
                                             , 'HW_platform': {'cpu_n': 12, 'threads_n': 24, 'gpu_n': 2}
                                             , 'layer-1': {'layer-2': 5}})

def test_parm_default():
    Paso(parameters_filepath='../parameters/test_base.yaml').startup()
    assert Param.parameters_D['project'] == 'Common Ground Solutions/paso'

def test_parm_file_error():
    with pytest.raises(ValueError):
        Paso('fred' ).startup()

def test_parm_startup_arg_error():
    with pytest.raises(TypeError):
        Paso().startup('fred')