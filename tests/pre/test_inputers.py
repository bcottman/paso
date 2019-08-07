import pandas as pd

import pytest

# paso imports

from paso.base import Paso, Param, PasoError
from paso.pre.inputers import Inputer, Splitter

fp = "../../parameters/default.yaml"
session = Paso(parameters_filepath=fp).startup()
# 0
def test_paso_param_file():
    assert session.parameters_filepath == fp


# 1
def test_paso_param_file_globa():
    assert session.parameters_filepath == Param.gfilepath


# 2
def test_sklearn_parm_global():
    assert Param.parameters_D["project"] == "Common Ground Solutions/paso"


# 3
def test_inputer_imputer():
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    assert inputer.inputers() == [
        "exec",
        "cvs",
        "xls",
        "xlsm",
        "text",
        "image2D",
        "image3D",
    ]


# 4
def test_inputer_transform_exec(flower):
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    assert (inputer.transform() == flower).any().any()


# 4b
def test_inputer_datasets():
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    assert inputer.datasets() == ["train"]


# 4c
def test_inputer_transform_exec_bad(flower):
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    assert (inputer.transform(dataset="test") == flower).any().any()


# 5
def test_inputer_transform_cvs_url():
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    names = [
        "Class",
        "Alcohol",
        "Malic-acid",
        "Ash",
        "Alcalinity-ash",
        "Magnesium",
        "phenols",
        "Flavanoids",
        "Nonflavanoid-phenols",
        "Proanthocyanins",
        "Color-intensity",
        "Hue",
        "OD280-OD315-diluted-wines",
        "Proline",
    ]

    winmeo = pd.read_csv(link, names=names).head()
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/wine.yaml")
    assert (inputer.transform().columns == winmeo.columns).any()


# 6
def test_inputer_transform_splitter_onto_wrong_place():
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    Flower = inputer.transform()
    splitter = Splitter()
    with pytest.raises(PasoError):
        train, valid = splitter.transform(
            Flower,
            target=inputer.target,
            ontological_filepath="../../ontologies/pre/inputers/split-stratify-shuffle-30.yaml",
        )


# 6b
def test_inputer_transform_splitter_onto():
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    Flower = inputer.transform()
    splitter = Splitter(
        ontological_filepath="../../ontologies/pre/inputers/split-stratify-shuffle-30.yaml"
    )
    train, valid = splitter.transform(Flower, target=inputer.target)
    assert train.shape[1] == 5


# 7
def test_inputer_transform_splitter_X_train():
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    Flower = inputer.transform()
    splitter = Splitter(
        ontological_filepath="../../ontologies/pre/inputers/split-stratify-shuffle-30.yaml"
    )
    train, valid = splitter.transform(Flower, target=inputer.target)
    assert valid.shape[1] == 5


# 8
def test_inputer_transform_splitter_X_test():
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    Flower = inputer.transform()
    splitter = Splitter(
        ontological_filepath="../../ontologies/pre/inputers/split-stratify-shuffle-30.yaml"
    )
    train, test = splitter.transform(Flower, target=inputer.target)
    assert test[train.columns.difference([inputer.target])].shape[1] == 4


# 8b
def test_inputer_transform_splitter_otto_group():
    inputer = Inputer(
        ontological_filepath="../../ontologies/pre/inputers/otto_group.yaml"
    )
    Flower = inputer.transform()
    splitter = Splitter(
        ontological_filepath="../../ontologies/pre/inputers/split-stratify-shuffle-30.yaml"
    )
    train, valid = splitter.transform(Flower, target=inputer.target)
    assert valid.shape[1] == 95


# 8c
def test_inputer_transform_splitter_wine():
    inputer = Inputer(ontological_filepath="../../ontologies/pre/inputers/wine.yaml")
    Flower = inputer.transform()
    splitter = Splitter(
        ontological_filepath="../../ontologies/pre/inputers/split-stratify-shuffle-30.yaml"
    )
    train, valid = splitter.transform(Flower, target=inputer.target)
    assert train.shape[1] == 14


# 9
def test_inputer_transform_ontological_arg_error(flower):
    o = Inputer()
    with pytest.raises(PasoError):
        Flower = o.transform(
            flower, ontological_filepath="../../ontologies/inputers/iris.yaml"
        )


# 10
def test_inputer_transform_ontological_bad_ontological_filepath(flower):
    o = Inputer(ontological_filepath="../../ontologies/inputers/XXXX.yaml")
    with pytest.raises(PasoError):
        Flower = o.transform()


# 11
def test_inputer_transform_ontological_flower(flower):
    o = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    Flower = o.transform()
    assert Flower.shape == flower.shape


# 12
def test_inputer_transform_wine():
    o = Inputer(ontological_filepath="../../ontologies/pre/inputers/wine.yaml")
    Wine = o.transform()
    assert Wine.shape == (178, 14)


# 13
def test_inputer_transform_ontological_otto_group():
    o = Inputer(ontological_filepath="../../ontologies/pre/inputers/otto_group.yaml")
    otto_group = o.transform()
    assert otto_group.shape == (61878, 95)


# 14

# 15
def test_inputer_train_ontological_arg_error():
    o = Inputer(ontological_filepath="../../ontologies/pre/inputers/iris.yaml")
    with pytest.raises(AttributeError):
        flower = o.traih()

# 16
def test_inputer_train_ontological_bad_file(flower):
    o = Inputer(ontological_filepath="../../ontologies/pre/inputers/bad.yaml")
    with pytest.raises(PasoError):
        _ = o.transform()

# 17
def test_inputer_pima_diabetes_train():
    o = Inputer(ontological_filepath="../../ontologies/pre/inputers/pima-diabetes.yaml")
    test = o.transform(dataset="train")
    assert test.shape == (768, 9)
# 18
def test_inputer_otto_group_test():
    o = Inputer(ontological_filepath="../../ontologies/pre/inputers/otto_group.yaml")
    test = o.transform(dataset="test")
    assert test.shape == (144368, 94)


# 19
def test_inputer_otto_groupsample_Submission():
    o = Inputer(ontological_filepath="../../ontologies/pre/inputers/otto_group.yaml")
    sampleSubmission = o.transform(dataset="sampleSubmission")
    assert sampleSubmission.shape == [144368, 10]
