
FAQ
===

What is **paso** standard documentation tool-set?
-------------------------------------------------

The documentation tools are:

::

    sphinx
    sphinx-apidoc
    sphinx-rtd-theme

What is **paso** standard test tool-set?
----------------------------------------

::

    pvtest
    coverage
    %timeit (in jupyter)
    dbg
    idbg (in jupyter)

What is **paso** development tool-set?
--------------------------------------

The development tools are:

::

    python 3.5 and up
    anaconda
    create and use environment using conda 
        suggest: environment named paso     
    install paso requirements
    (now conda activate paso)
    jupyter (via anaconda)
    pycharm
    git
    github

What is the difference between the ``pandas.DataFrame`` and ``paso.common.toDataFrame`` ?
-----------------------------------------------------------------------------------------

``paso.common.toDataFrame`` enables **paso** to add additionality
functionality without changing the ``pandas.DataFrame`` defintion.
Please use with **paso** ``paso.common.toDataFrame``.

What is the difference between the scikit and the **paso** ?
------------------------------------------------------------

-  **Standard input and output is a pandas dataframe**

   -  sckit-learn standard input and output is a numpy array

-  **paso** new data objects with a 2-phase ``train/predict`` as models
   (``pasoModel``).
-  **paso** creates new data objects with a 1-phase ``transform`` as
   functions (``pasoFunction``).

What is the difference between the sci-kit pipeline and the **paso** pipeline?
------------------------------------------------------------------------------

In a pipeline a series of data-operations are connected together. In a
sci-kit data-analytics pipeline, this is usually a left-to-right series
of calls of transformer calls ended by an estimator (model), that is a
PIPELINE of invocations.

What **paso** offers with ``pasoLine`` (sic. paso pipeline) is
everything that sci-kit does (in fact, **paso** can be used with
sci-kit) , plus:

1. one or more named event loggers;
2. check-pointing by save/load of models;
3. persisting by read/write transformed datasets;
4. flexible parameter setting and reading;
5. use of any available GPUs;
6. multi-threaded or multi-process for multi-core CPUs.
7. scaffolding for the hoped-for emergence of quantum computing.

Error "no module named [some module]"
-------------------------------------

In the repository folder do:

::

    pip install -r requirements.txt 

this should solve this error.

Are there tutorials for **paso** ?
----------------------------------

Yes. In the form of jupyter notebooks. See
`lessons <https://github.com/bcottman/paso/paso/lessons>`__

Is **paso** Threadsafe?
-----------------------

**paso**, itself, is thread-safe. However, you will use **paso** with
many other packages which may not be thread-safe. For example, numpy is
thread-safe. ndarrays can be accessed in a thread-safe manner, but you
must be careful with state if you mutate an array.

In Pandas, deleting a column is usually not thread-safe as changing the
size of a DataFrame usually results in a new Dataframe object. Any
**pasoFunction** with ``iplace=True`` will not be thread-safe.

At some point this may change in Pandas and other python libaries as
multi-cored CPUs are becoming common.
