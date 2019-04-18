import pytest
#
from paso.base import pasoFunction, pasoModel,pasoError,pasoLine
from paso.base import dask_pandas_ratio,_read_paso_parameters, pasoLine,get_paso_log
from paso.base import _start_paso_log, get_paso_log, _flush_paso_log
import logging
#from attrdict import AttrDict


def test_pasoline_get_logger():
#    pipy = pasoLine()
    assert type(get_paso_log()) == logging.Logger

def test_pasoline_startup_return():
    pipy = pasoLine()
    assert type(pipy.startup()) ==  type(pipy)

def test_pasoline_startup_parameter_blank_filename():
    pipy = pasoLine()
    assert pipy.startup() ==  pipy

def test_pasoline_log_property():
    pipy = pasoLine()
    pipy.log = 'RED'
    assert type(get_paso_log(pipy.log)) == type(get_paso_log('RED'))

def test_read_parameters():
    assert _read_paso_parameters('../lessons/parameters/lesson.1.yaml')['threads_n'] == 24

def test_pasoLine_startup_parameters():
    p = pasoLine( parameter_filepath='../lessons/parameters/lesson.1.yaml').startup().parameters
    assert p['threads_n'] == 24

def test_pasoP_startup_default_parameters():
    p = pasoLine().startup().parameters
    assert p['threads_n'] == 24

def test_global__PiPeLiNeS__and_startup():
    assert pasoLine().startup().show()[0][1] == 'square'

def test_default_log_name():
    assert pasoLine().log_name == 'paso'

def test_log_name():
    assert pasoLine(log_name='xyz').log_name == 'xyz'

def test_default_log_instance_without_startup():
    assert pasoLine().log == None

def test_default_log_instance_with_startup():
    assert pasoLine().startup().log == get_paso_log()

def test_global__PiPeLiNeS__and_shutdown():
    assert pasoLine().startup().shutdown().show()[1][2] == ' '

def test_pasoFunction_train_properties():
    ins = pasoFunction()
    assert ins.trained == None

def test_pasoFunction_all_properties():
    ins = pasoFunction()
    assert ( not ins.trained\
           and not ins.transformed \
           and not ins.persisted\
            and not ins.cache) == True

def test_pasoFunction_f_x_None():
    ins = pasoModel()
    assert ins.f_x == None

def test_pasoFunction_train_not_implemented():
    ins = pasoFunction()
    with pytest.raises(AttributeError):
        ins.train()

def test_pasoFunction_predict_not_implemented():
    ins = pasoFunction()
    with pytest.raises(AttributeError):
        ins.predict()

def test_pasoFunction_transform_not_implemented():
    ins = pasoFunction()
    with pytest.raises(NotImplementedError):
        ins.transform()

def test_pasoFunction_save_not_implemented():
    ins = pasoFunction()
    with pytest.raises(AttributeError):
        ins.save('nofilepath')

def test_pasoFunction_load_not_implemented():
    ins = pasoFunction()
    with pytest.raises(AttributeError):
        ins.load('nofilepath')

def test_pasoFunction_persist_before_transform_error():
    ins = pasoFunction()
    with pytest.raises(pasoError):
        ins.write('nofilepath')

def test_pasoFunction_train_required():
    ins = pasoModel()
    assert (not ins.trained) == True

def test_pasoFuntion_caceheOn(df_City):
    ins = pasoFunction().cacheOn()
    assert (not ins.trained\
            and not ins.persisted \
            and (ins.save_file_name == '') \
            and not ins.model_cache
            and ins.cache)  == True

def test_pasoFuntion_caceheOff(df_City):
    ins = pasoFunction().cacheOn().cacheOff()
    assert (not ins.trained\
            and not ins.persisted \
            and (ins.save_file_name == '') \
            and not ins.model_cache
            and not ins.cache)  == True


def test_pasoModel_all_properties():
    ins = pasoModel()
    assert not ins.trained\
           and not ins.persisted \
                and (ins.save_file_name == '')\
                 and not ins.transformed \
                 and not ins._cache  == True

def test_pasoModel_caceheOn():
    ins = pasoModel().cacheOn()
    assert not ins.trained\
           and not ins.persisted \
                and (ins.save_file_name == '')\
                 and not ins.trained \
                 and ins.cache  == True

def test_pasoModel_caceheOnTrain_not_implemented(df_City):
    ins = pasoModel().cacheOn()
    with pytest.raises(NotImplementedError):  # to be implemented in specfic
        ins.train(df_City)
        assert not ins.trained\
               and not ins.persisted \
                    and (ins.save_file_name == '')\
                     and ins.transformed \
                     and ins.cache  == True

def test_pasoModel_caceheOn(df_City):
    ins = pasoModel().cacheOn()
    assert (not ins.trained\
            and not ins.persisted \
            and (ins.save_file_name == '') \
            and not ins.model_cache
            and ins.cache)  == True

def test_pasoModel_Model_caceheOn():
    ins = pasoModel().model_cacheOn()
    assert (not ins.trained\
            and not ins.model_persisted \
            and (ins.save_model_file_name == '')\
            and ins.model_cache
            and not ins.cache)  == True

def test_pasoModel_both_caceheOn():
    ins = pasoModel().model_cacheOn().cacheOn()
    assert (not ins.trained\
            and not ins.model_persisted \
            and (ins.save_model_file_name == '')\
            and ins.model_cache
            and ins.cache)  == True

def test_pasoModel_reset():
    ins = pasoModel().cacheOn().reset()
    assert (not ins.trained \
           and not ins.persisted \
           and (ins.save_file_name == '') \
           and not ins.trained \
           and not ins.cache) == True

def test_pasoFunction_reset_badFile():
    ins = pasoFunction().cacheOn()
    ins.save_file_name = 'foooy'
    ins.persisted = True
    with pytest.raises(NameError):
        ins.reset()

def test_pasoModel_reset_badFile():
    ins = pasoModel().cacheOn()
    ins.save_file_name = 'foooy'
    ins.persisted = True
    with pytest.raises(NameError):
        ins.reset()

def test_pasoModel_f_x_None():
    ins = pasoModel()
    assert ins.f_x == None

def test_pasoModel_train_not_implemented(X):
    ins = pasoModel()
    with pytest.raises(NotImplementedError):
        ins.train(X)

def test_pasoModel_predict_not_implemented(X):
    ins = pasoModel()
    with pytest.raises(NotImplementedError):
        ins.predict(X)

def test_pasoModel_transform_not_implemented():
    ins = pasoModel()
    with pytest.raises(AttributeError):
        ins.transform()

def test_pasoModel_save_bad_file():
    ins = pasoModel()
    with pytest.raises(pasoError):
        ins.save('nofilepath')

def test_pasoModel_load_bad_file():
    ins = pasoModel()
    with pytest.raises(NameError):
        ins.load('nofilepath')

# this test requires 50 sec or more clock time
def test_dask_pandas_ratio_No_Args():
    assert dask_pandas_ratio().loc[0, 0] == 'count'