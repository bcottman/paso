from setuptools import setup,find_packages

setup(name='paso',
      packages=find_packages(),
      version='0.3.0',
      description='A python package for data wrangling for models.',
      long_description= """
**paso** is a set of function and class method operations for data wrangling Machine
for Learning and Deep Learning models.
**paso** is Spanish for the English word step. A **paso** class or function is a step in your
analysis pipeline. Because **paso** follows a protocol, the set of all operations in **paso** can be
called a framework.

Data engineers and Machine Learning Scientists appreciate the value of having their tools span
across the data science workflow. The framework **paso** has been designed to fit in such all
encompassing frameworks, as mlflow and Berkeley Data Analytics Stack(BDAS). It has been designed for
pre-processing and post-processing data wrangling to and from learning packages such as PyTorch,
Tensorflow, XGBoost and scikit-learn. It has been designed to work with a large number of evolving
visialization tools such as facet, matplotlib, seaborn, Tensorboard, and plotly, to name just a few.

You can contribute to the **paso** Project in many ways. Below are listed some of the areas:

-  Fix a typo(s) in documentation.
-  Improve some of the documentation.
-  Fix typo or/and improve a docstring(s).
-  Report a documentation bug.
-  Improve existing test or/and code more tests.
-  Execute test suite for a distribution candidate and report any test failures.
-  Post a new issue.
-  Fix a bug in a issue.
-  Re-factor some code to add functionality or improve performance.
-  Suggest a new feature.
-  Implement a new feature.
-  Improve or/and add a lesson.

Remember to post in issues the proposed change and if accepted it will be closed. See
`issues <https://github.com/bcottman/paso/issues>`__ or
`projects <https://github.com/bcottman/paso/projects/1>`__ for open issues for the **paso** project.

You can find a issue , or you can post a new issue, to work on. Next, or your alternative first
step, you need to set-up your local **paso** development environment. Code, Documentat
""",
      url='https://github.com/bcottman/paso',
      download_url='https://github.com/bcottman/paso/archive/0.2.8.tar.gz',
      author='Bruce Cottman',
      author_email='dr.bruce.cottman@gmail.com',
      keywords=['machine-learning', 'deep-learning', 'transformations', 'functional','data-analytics'],
      copyright = 'Copyright (c) 2018 Bruce H. Cottman',
      license='MIT',
      install_requires=[
          'attrdict>=2.0.1',
          'ipython>=6.4.0',
          'numpy>=1.14.0',
          'numba>=0.40.0',
          'fancyimpute>= 0.4.1',
          'tqdm>= 4.28.1',
          'pandas>=0.22.0',
          'pyarrow>=0.11.1',
          'pydot_ng>=1.0.0',
          'pytest>=3.6.0',
          'scikit_learn>=0.19.0',
          'scipy>=1.0.0',
          'setuptools>=39.2.0',
          'typing>=3.6.4'
          'yaml>=0.1.7'],
      zip_safe=False,
      classifiers=[])
