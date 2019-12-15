# paso changelog

## 0.5.2

2019-11-17

### Added

- add type hints
- add doc Learners
- refactor Encoders
- add Encoders.category_encoder_embedded
- add Learner.tune_hyperparameters
- add to_util.py and refactor to function from class toX
- add method call to dataframe for cleaners and to_util.py
- add selected pyanitor functions to clean

## 0.5.1

2019-11-07

### Added
- refactored Cleaners
- added to cleaner test
- added cross-validation test
- added lesson 2


## 0.5.0

2019-08-28

### Added

- changed ontology to description
- refactored inputers.py
- added to test_inputers.py
- refactored cleaners.py
- added transform_booleans in cleaners.py
- added to test_cleaners.py
- refactored learners.py
- added cross-validation to learners.py
- added predict, predicta, evaluate learners.py
- added hyperopt for hyperparamer tuning to learners.py
- added to test_learners.py
- refactored encoders..py
- added to test_encoders.py
- finished lesson-3.ipynb

## 0.4.2

2019-08-28

### Added

- updated lesson-3.ipynb
- added lesson-4.ipynb
- added lesson-5.ipynb
- fixed doc bugs
- fixed .py bugs

## 0.4.1

2019-08-25

### Added

- added tests/test_inputers
- added tests/test_learners
- added to tests/test_cleaner
- standardize into uniform
- refactored .py for new ontology structures
- fixed bugs in all classes
- changed cross_validators into Learner post-operations
- added Learner/classification metrics
- added Plotter class
- added Balancer class
- added Augmenter class

## 0.4.0

2019-07-27

### Added

- added ontologies
- added tests/test_inputers
- added tests/test_learners
- added to tests/test_scalers
- added to tests/test_cleaner
- added inputers
- added exec
- added cvs
- added splitter
- added __cross_validators__
- added BaggingClassifierCV
- bagging with or wihout sample replacement
- added CalibratedClassifierCV
- added __learners__
- added LinearRegression
- added RandomForestClassifier

## 0.3.2

2019-07-14

### Added

- lesson-1.ipynb
- logging-parameter.ipynb
- lesson-2.ipynb
- lesson-3.ipynb
- added inputers
- added learners/rf
- updated scalers
- changed base to be synched with lesson-1
- changed cleaning to be synched with lesson-2
- added impute to clearners
- renamed EliminateUnviableFeatures.py to cleaning.py
- Param - eliminated need for bootstrap file

## 0.3.1

2019-06-07

### Added

- added Encoder

## 0.3.0

2019-06-05

### Added

- added Scaler

## 0.2.1.alpha

2019-04-28

### Added

- reran all tests
- created package at pypi

## 0.2.0

2019-04-10

### Added

- refactor with decorators ModelWrap and FunctionWrap

### Added

- decorator class pasoDecorators
- decorator TransformWrap

## 0.1.0

2018-01-10

### Added

- base for pypi package
- base paso/pre
- base tests
- paso/docs/nbdoc
- `chi_squared_dist_table()` - A table of χ2 values vs p-values
- `bernoulli()` - the Bernoulli distribution
- `poisson()` - the Poisson distribution