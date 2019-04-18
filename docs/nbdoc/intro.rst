
What is paso?
=============

**paso** is a set of function and class methods operations for Machine
Learning and Deep Learning data munging. **paso** is Spanish for the
English word step. A **paso** class or function is a step in your
analysis pipeline. Because **paso** follows a protocol, the set of all
operations in **paso** can be called a framework.

Data engineers and Machine Learning Scientists appreciate the value of
having their tools span across the data science workflow. The framework
**paso** has been designed to fit in such all encompassing frameworks,
as mlflow and Berkeley Data Analytics Stack(BDAS). It has been designed
for pre-processing and post-processing data wrangling to and from
learning packages such as PyTorch, Tensorflow, XGBoost and scikit-learn.
It has been designed to work with a large number of evolving
visialization tools such as facet, matplotlib, seaborn, Tensorboard, and
plotly, to name just a few.

What problem does **paso** solve?
=================================

**IN THE BEGINNNG..**, **paso** did not even have a name. It was a
gaggle of various worker-bee functions for cleaning and munging data
into a form that could be used by various machine learning models in
``scikit-learn``. Then it started to get a bigger and a more complex set
of code mainly driven by different **kaggle** datasets.

As we evolved this code base, we were heavily influenced by:

-  an almost overwhelming bounty of well-maintained packages in the
   python ecosystem;
-  perhaps the biggest, if not one of the biggest active community of
   open-source developers;
-  seemingly the best python software was open-source;
-  the best machine learning and deep learning research, papers and
   software were and still are in the public domain;
-  we were adding on common code for repeatable functionality, that some
   great target package did not have (yet!).
-  we will and are turning off services that are better (IOHO) in other
   packages.

Thus was conceived **paso**, if you will, not a framework, but a
meta-framework, for using various python packages. **paso** is a common
coding approach with common functionality:

-  pandas dataframe are the common **paso** data objects, while
   scikit-learn's common data object is the numpy array.

   -  most types; list, tuple, numpy array, pandas DataFrame or pandas
      series are converted to **paso** using the function
      ``paso.base.toDataFrame``.

   -  Input to any **paso** must be a pandas DataFrame. Thus the
      function ``paso.base.toDataFrame`` to covert most types.

   -  Output is a **paso** DataFrame or ``None``.

-  logging action have occurred, errors at the **paso** level, and
   debugging;

   -  **paso** has a common logging object. Each log instance can be
      named, allowing multiple logger instances. This is essentially
      free for us as we are able to use the wonderful **logger**
      package.

-  check-pointing the model during long, multi-step training and
   prediction runs;

   -  **paso** has common save and load methods for all models in the
      ``pasoModelBase`` class.

-  check-pointing the transformed data objects;

   -  **paso** has common read and write methods for all transformers in
      the ``pasoFunctionBase`` class.

-  difficulties with reproducing experimental results;

   -  All model, encoders, functions, transformers are encapsulated as
      the\ ``pasoModelBase`` class or the ``pasoFunctionBase`` class.
      Continual bug-fixing, documentation, memory efficiency, security,
      and performance improvements can occur without effecting the data
      scientist.

-  difficulties with creating different experiments using parameter
   change;

   -  **paso** has a common file path, file structure, reading-in, and
      setting for any parameter from one experiment-dependent common
      file. Both standard as well as experiment-specfic parameters can
      be assigned. Also, each parameter instance is named, allowing
      multiple parameter instances, should you find you want more than
      one. Again, we see the benefit of using the mature **yaml**
      package.

-  adding new state-of-the-art or trial models or transformations;

   -  the\ ``pasoModelBase`` class or the ``pasoFunctionBase`` class can
      encapsuate usually anything such that new fuctionality is within
      the **paso** protocol space. Indeed, most functionality from
      **sklearn, scipy, augment, keras, pytorch**, to name just a few,
      are wrapped into the **paso** space. In a few cases, new code is
      used where functionality does not exist yet, or better performance
      is needed for larger data sets.

Installation
============

**paso** requires ``python3.5`` or above.

::

     pip install paso

(Recommended) installation: Create and use Anaconda virtual environment
-----------------------------------------------------------------------

Create a new anaconda virtual environment or use any other virtual
environment tool of your choosing. (we use **paso** in this example).
Set yourself in the anaconda virtual environment: **paso**

::

    conda activate paso
    (paso) Mac-Pro:~ home$ pip install paso

Other virtual environment systems can be used, but just Anaconda
examples are shown.

Feature Requests
================

Please send us your ideas on how to improve the **paso** package. We are
looking for your comments: `Feature
requests <https://github.com/bcottman/paso/issues>`__.

Resources
=========

1. `Documentation <https://paso.readthedocs.io/en/latest>`__
2. `Source <https://github.com/bcottman/paso>`__
3. `Bugs reports <https://github.com/bcottman/paso/issues>`__
4. `Feature requests <https://github.com/bcottman/paso/issues>`__
5. Lesson notebooks:

   -  `Lesson 1: Quick
      start <https://github.com/bcottman/paso/lessons/lesson-1.ipynb>`__
   -  ...

Directory structure
===================




