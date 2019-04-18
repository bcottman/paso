from pathlib import Path

import pytest
# paso imports

from paso.base import pasoFunction,pasoError,_Check_No_NA_Values,get_paso_log
from paso.base import is_DataFrame,toDataFrame
from paso.pre.scale import scaler
#0
def test_df_Class_init_NoArg():
    with pytest.raises(TypeError):
        g = scaler()
#1
def test_df_Class_init_WrongScaler():
    with pytest.raises(pasoError):
        g = scaler('GORG')
# BoxCoxScaler unit tests
#2
def test_df_BoxCox_City_negative_error(df_City):
    g = scaler('BoxCoxScaler')
    with pytest.raises(ValueError):
        g.train(df_City)
#3
def test_df_BoxCox_2d_numpy(X):
    X = toDataFrame(X)
    g = scaler('BoxCoxScaler')
    g.train(X)
    assert g.predict(X).shape == X.shape
#4
def test_df_BoxCox_instance_1d(y):
    g = scaler('BoxCoxScaler')
    assert g.train(y) == g
#5
def test_df_BoxCox_1d_numpy(y):
    g = scaler('BoxCoxScaler')
    g.train(y)
    assert g.predict(y).squeeze().shape ==y.shape
#6
def test_df_BoxCox_type_error(ystr):
    g = scaler('BoxCoxScaler')
    with pytest.raises(TypeError):
        g.train(ystr())
# LambertScaler unit tests
#7
def test_df_Lambert_instance_2d(X):
    g = scaler('LambertScaler')
    assert g.train(X) == g
#8
def test_df_Lambert_2d_numpy(X):
    g = scaler('LambertScaler')
    g.train(X)
    assert g.predict(X).shape == X.shape
#9
def test_df_Lambert_instance_1d(y):
    g = scaler('LambertScaler')
    assert g.train(y) == g
#10
def test_df_Lambert_1d_numpy(y):
    g = scaler('LambertScaler')
    g.train(y)
    assert g.predict(y).squeeze().shape == y.shape
#11
g = scaler('LambertScaler')
def test_df_Lambert_type_error(ystr):
    g = scaler('LambertScaler')
    with pytest.raises(TypeError):
        g.train(ystr)
#12
def test_df_Lambert_zeroval_1d(yz):
    g = scaler('LambertScaler')
    g.train(yz)
    assert g.predict(yz).squeeze().shape == yz.shape
#13
def test_df_Lambert_no_fit(yn):
    g = scaler('LambertScaler')
    with pytest.raises(AttributeError):
        g.fit()
#14
def test_scalerList(X):
    g = scaler('BoxCoxScaler')
    assert g.scalers() == ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer', 'BoxCoxScaler', 'LambertScaler']
#15
def test_df_BoxCox_inverse(y):
    g = scaler('BoxCoxScaler')
    z = g.train(y,inplace = False)
    assert (g.inverse_predict(g.predict(y) == y).any())
#16
def test_df_Lambert_save(yn):
    g = scaler('LambertScaler').reset().cacheOn()
    g.train(yn).predict(yn)
    fp: str = 'tmp/df'
    g.save(fp)
    assert (g.trained and g.predicted  and g.cache and g.persisted and (g.save_model_file_name ==  fp)) == True
#17
def test_df_Lambert_write(df_City):
    g = scaler('LambertScaler').reset().cacheOn()
    g.train(df_City).predict(df_City)
    fp: str = 'tmp/df_write'
    g.write(fp)
    assert (g.trained and g.predicted  and g.cache and g.persisted and (g.save_file_name == fp)) == True
#18
def test_df_Lambert_negval_1d(yn):
    g = scaler('LambertScaler')
    g.train(yn)
    assert g.predict(yn).squeeze().shape == yn.shape
#19
def test_bad_scale_name():
    with pytest.raises(pasoError):
        g = scaler('fred')