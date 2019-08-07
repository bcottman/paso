
Contributing to any of the **paso** Projects
============================================

The public `project <https://github.com/bcottman/paso>`__ is on GitHub .

You can contribute to the **paso** Project in many ways. Below are
listed some of the areas:

-  Fix a typo(s) in documentation.
-  Improve some of the documentation.
-  Fix typo or/and improve a docstring(s).
-  Report a documentation bug.
-  Improve existing test or/and code more tests.
-  Execute test suite for a distribution candidate and report any test
   failures.
-  Post a new issue.
-  Fix a bug in a issue.
-  Re-factor some code to add functionality or improve performance.
-  Suggest a new feature.
-  Implement a new feature.
-  Improve or/and add a lesson.

Remember to post in issues the proposed change and if accepted it will
be closed. See `issues <https://github.com/bcottman/paso/issues>`__ or
`projects <https://github.com/bcottman/paso/projects/1>`__ for open
issues for the **paso** project.

You can find a issue , or you can post a new issue, to work on. Next, or
your alternative first step, you need to set-up your local **paso**
development environment. Code, Documentation and Test can have separate
or shared environments. Each one is discussed in the following sections.

Contributing to the **paso** Documentation
------------------------------------------

One of the best ways to start CONTRIBUTE-ing to the **paso** project is
by accomplishing a documentation task.

The **paso** project uses ``Sphinx``. Sphinx, written by Georg Brandl
and licensed under the BSD license, was originally created for the
Python documentation. Unfortunately, you will be wresting with ``.rst``
unless you choose the much simpler markup of Jupyter notebooks or even
the more powerful ``LaTex``.

Once you finish your documentation task, please Submit a GitHub pull
request, as show in

::

    Creating a push request 

in `Github pull
request <https://github.com/bcottman/paso/tree/master/docs/nbdoc/Contributing.ipynb>`__

Documentation sources can be found `Doc
sources <https://github.com/bcottman/paso/docs/nbdoc/>`__

**paso** Documentation can be found
`Docs. <https://paso.readthedocs.io>`__

Create a Documentation environment using conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of what you need in both ``Sphinx`` packages and environment
creation are in the `Anaconda <https://www.anaconda.com>`__
distribution.

First you should create an environment for **paso** documenation.
Detailed instuctions are found
`here. <https://conda.io/docs/user-guide/tasks/manage-environments.html>`__.

Almost all of the packages you need for documentation are included in
the Anaconda distribution. You only need add to your environment (for
this and other examples we call our environment **paso**).

::

    >>> (paso) pip install paso
    >>> (paso) pip install sphinx
    >>> (paso) pip install sphinx-rtd-theme 
    >>> (paso) pip install sphinx-apidoc

Create your own docs
~~~~~~~~~~~~~~~~~~~~

The ``paso project has created the``\ docs\`\`directory and
the initial infrastructure. You will finish creating your local doc
development environment.

Clone the
``paso to the local directory you have selected using``\ git\`\`,

Go topaso project directory

::

    >>> (paso) cd .../paso/docs

Create ``rst`` files from docstrings in ``py`` files.:

::

    >>> (paso) sphinx-apidoc -o source/ -d 3 -fMa ../paso

Generate documnentation:

::

    >>> (paso) clear;make clean;make html

The HTML of the documentation can be found
`here. <.../paso/docs/_build/html/index.html>`__

How to use notebooks (``ipynb`` ) for documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python notebooks (``ipynb`` ) can be used to document instead of or with
``rst`` files. ``nbsphinx`` is a Sphinx extension that enables ``ipynb``
files.

To Install ``nbsphinx``:

::

    >>> (**paso**) pip install nbsphinx --user

in your **paso** doc environment.

Edit your ``conf.py`` and change ``source_suffix``:

::

    source_suffix = ['.rst', '.ipynb']

Edit your ``index.rst`` and add the names of your \*.ipynb files to the
toctree.

More detailed information is found
`here. <https://nbsphinx.readthedocs.io/en/0.2.8/rst.html>`__

Sphinx and other Documentation Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Overview <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`__
-  `yao <https://codeandchaos.wordpress.com/2012/07/30/sphinx-autodoc-tutorial-for-dummies/>`__

Contributing to the **paso** Issues and Reviews
-----------------------------------------------

-  Answering queries on the issue tracker.
-  Investigating, isolating or/and implementing code that demostrates a
   bug that other(s) have reported on
   `issues <https://github.com/bcottman/paso/issues>`__.
-  Review other developers’ pull requests especially merges that could
   cause conflicts.
-  Report any issues that you may encounter.
-  Reference the project from your blog aor publications that you may
   write.
-  Link to it from your social media account or website
-  Use your imagination. Let us know as we will add to this list.

Contributing to the **paso** Tests
----------------------------------

The next suggested step (sic.) to **CONTRIBUTE** to the various **paso**
projects is testing. Create a testing enhancement, then Submit a GitHub
pull request.

Test sources can be found `**paso** unit test
suite <https://github.com/bcottman/**paso**/tests>`__. Developing a
`lesson <https://github.com/bcottman/paso/paso/lessons>`__ serves also
as integration test.

Adding more tests for existing **paso** objects, and other supporting
code is a great method to familiarize yourself and make your starting
contributions to the **paso** project.

Also,it will be not be possible for your contributed code to be merged
into the master **paso** repo without accompanying docstring and unit
tests that provide coverage for the critical parts of your contribution.

You can expect your contribution to not past review unless tests are
provided to cover edge cases and test for error conditions. Remember,
you are asking people to use your contributed code.

**peso** Test Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

Some of these guidelines have been adapted from `writing
tests <https://docs.python-guide.org/writing/tests/>`__ and `pandas
testing <https://github.com/pandas-dev/pandas/wiki/Testing>`__.

-  (RECCOMENED) Learn your tools and learn how to run a single test or a
   test case. Then, when developing a function inside a module, run this
   function’s tests frequently, ideally automatically when you save the
   code.

-  (REQUIRED) Each test unit must be fully independent. Each test must
   be able to run alone, and also within the test suite, regardless of
   the order that they are called. The implication of this rule is that
   each test must be loaded with a fresh dataset and may have to do some
   cleanup afterwards. This standard is that this is handled by setUp()
   and tearDown() methods. (if you use ``pytsest``\ it will take care of
   this you.)

-  (RECCOMENED) Run the full test suite before a coding session, and run
   it again after.

-  (RECCOMENED) The first step when you are debugging your code is to
   write a new test pinpointing the bug. While it is not always possible
   to do, those bug catching tests are among the most valuable pieces of
   code in your project.

-  (RECCOMENED) Use long and descriptive names for testing functions.
   These function names are displayed when a test fails, and should be
   as descriptive as possible.

-  (REQUIRED) When something goes wrong or has to be changed, and if
   your code has a good set of tests, you or other maintainers will rely
   largely on the testing suite to fix the problem or modify a given
   behavior. Therefore the testing code will be read as much as or even
   more than the running code.

-  (RECCOMENED) Testing code is as an introduction to any developers.
   When someone will have to work on the code base, running and reading
   the related testing code is often the best thing that they can do to
   start. They will or should discover the hot spots, where most
   difficulties arise, and the corner cases. If they have to add some
   functionality, the first step should be to add a test to ensure that
   the new functionality is not already a working path that has not been
   plugged into the interface.

Create a testing environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend you create a virtual environment for your testing. Use your
favorite tool to create a virtual environment.

Use or source activate (on mac or ubuntu) the virtual environment named
paso:

::

    >>> source activate paso

install the packages you will need to develop test for paso. The
following are the standard packages we use:

::

    (paso)>>> pip install paso 
    (paso)>>> pip install pytest
    (paso)>>> pip install pandas
    (paso)>>> pip install coverage

You may already have pandas as part of your environment. What you will
need to import into python is:

::

    import pytest
    # paso imports
    import joblib
    from paso.pasoBase import pasoFunctionBase, pasoModelBase,pasoError
    from paso.common.PipeLine import get_paso_log
    <any other needed paso files>

Recommended Testing Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  https://docs.python-guide.org/writing/tests/
-  https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest
-  http://pythontesting.net/framework/nose/nose-introduction/
-  https://ymichael.com/2014/12/17/python-testing-with-nose-for-beginners.html
-  https://github.com/pandas-dev/pandas/wiki/Testing

Coverage tool for **paso**
~~~~~~~~~~~~~~~~~~~~~~~~~~

Coverage measurement is typically used to measure the effectiveness of
tests. It can show which parts of your code are being exercised by
tests, and which are not. You can use any coverage tool you wish. We
recommend

::

    Coverage.py (see documentation for installation and usage) 

,a tool for measuring code coverage of Python programs. It monitors your
test suite, noting which parts of the code have been executed, then
analyzes the source to identify code that could have been executed but
was not.

Also a good introduction to Coverage.py is:

::

    https://www.blog.pythonlibrary.org/2016/07/20/an-intro-to-coverage-py/

Branch Coverage
^^^^^^^^^^^^^^^

You can use another powerful feature of coverage.py: branch coverage.
Testing every possible branch path through code, while a great goal to
strive for, is a secondary goal to getting 100% line coverage for the
entire **paso** package.

If you decide you want to try to improve branch coverage, add the
``--branch`` flag to your coverage run:

::

    /python COVERAGEDIR run --pylib --branch <arguments>

This will result in a report stating not only what lines were not
covered, but also what branch paths were not executed.

Contributing to **paso** Distributions
--------------------------------------

This contribution consists of running the test suite on configuration of
the underlying environment of the distribution.

Contributing to the **paso** Code
---------------------------------

1. You want to propose a new Feature and implement it post about your
   intended feature (under issues or projects) and project management
   shall discuss the design and implementation. Once we agree that the
   plan looks good, go ahead and implement it.

2. You want to implement a feature refactor or bug-fix for an
   outstanding issue. Look at the outstanding
   `issues <https://github.com/bcottman/paso/issues>`__. Pick an issue
   and comment on the task that you want to work on this feature. If you
   need more context on a particular issue, please ask and someone will
   provide a suggestion on how to proceed.

Code sources can be found under the `source
code <https://github.com/bcottman/paso/paso>`__/ directory.

Once you finish your feature enhancement, please Submit a GitHub pull
request.

Contributing to the **paso** lessons
------------------------------------

**paso** enables both documentation and learning by **paso**
**lessons**. Tasks for **lessons** include:

1. Add to/improve a lesson.
2. Implement new lesson.

Creating a push request
-----------------------

Navigate to the `paso repo <https://github.com/bcottman/paso>`__.

1. Click the “Fork” button in the top-right part of the page, right
   under the main navigation. A copy of the **paso** repo in your Github
   account.
2. Clone your Github account copy of the of the **paso** repo on your
   local client machine where you will do your development enhancements
   for the **paso** project.

   ::

       cd <local client  machine development directory>
       git clone https://github.com/<your github username>paso.git

Locally, create a branch for your work

::

    git checkout -b <branch-name>  

Locally, accomplish your changes to N files named <file-1>...<file-n>

::

    git add <file-1>
    .
    .
    git add <file-n> 

commit locally N-files from staging area

::

    git commit -a -m '<message-documentation-change>'

show associated remote repo on GitHub

::

    git remote -v

push to remote GitHub, aliased as origin, from local branch
<branch-name>

::

    git push origin <branch-name>

1. on your remote Github account repo, change to branch

2. Navigate to the base repo at `paso
   repo <https://github.com/bcottman/paso>`__ issues and click the “New
   Pull Request” button

what you are doing is: “I have some code in my fork of the
\*\*\*\*paso\*\*\*\* project in ``<branch-name>`` that I want to merge
into the \*\*\*\*paso\*\*\*\* base repo
