import pandas as pd
import numpy as np
import pytest

from paso.base import PasoError, Param, Paso,Log

def test_paso_Class_init_BadArg():
    with pytest.raises(TypeError):
        g = toDataFrame(fred=1)
#2
def test_paso_2d_numpy_labels_(X,cn):
    g = toDataFrame()  #
    assert g.transform(X,labels= cn).columns[1] == "k"
#3
def test_paso_2d_numpy_labels_default(X, cn):
        g = toDataFrame()  #
        assert g.transform(X).columns[1] == "c_1"
#4
def test_paso_1d_numpy_bad_arg(y,cn):
    g = toDataFrame()
    with pytest.raises(PasoError):
        g.transform(y)
#5
def test_paso_2d_named_too_many_args(X, cnv):
    g = toDataFrame()
    with pytest.raises(ValueError):
        g.transform(X,cnv,1).columns[2] == cnv[2]
#6
def test_paso_2d_named(X,cnv):
    g = toDataFrame()
    assert g.transform(X,labels= cnv).columns[2] == cnv[2]

# edge tests
def test_paso_Blank_Arg_Transform():
    g = toDataFrame()
    with pytest.raises(TypeError):
        g.transform()
#8
def test_paso_Empty_list_Arg_Transform():
    g = toDataFrame()
    with pytest.raises(PasoError):
        g.transform([])
#9
def test_paso_3dArray_numpy(X):
    g = toDataFrame()
    with pytest.raises(ValueError):
        g.transform(np.reshape(X,(-1,-1,1)))
#10
def test_paso_df_labels_set(X,cnv):
    g = toDataFrame()
    assert (g.transform(pd.DataFrame(X,columns=cnv),labels = cnv).columns == cnv).all()
#11
def test_toDataFrame_save_not_implemented():
    ins = toDataFrame()
    with pytest.raises(AttributeError):
        ins.save('nofilepath')
#12
def test_toDataFrame_load_not_implemented():
    ins = toDataFrame()
    with pytest.raises(AttributeError):
        ins.load('a foobar')
#13
def test_toDataFrame_write_before_transform_error():
    ins = toDataFrame()
    with pytest.raises(PasoError):
        ins.write('nofilepath')
#14
def test_toDataFrame_write_before_transform_error(X):
    ins = toDataFrame()
    ins.transform(X)
    assert ins.write('tmp/X.txt') == True

    # 14
def test_toDataFrame_read_after_write(X):
    ins = toDataFrame()
    ins.transform(X)
    if ins.write('tmp/X.txt'):
        assert (ins.read('tmp/X.txt') == X).all().all()
    esle: False
