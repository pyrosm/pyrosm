Contributing to pyrosm
======================

Contributions of any kind to pyrosm are more than welcome. That does not mean
new code only, but also improvements of documentation and user guide, additional
tests (ideally filling the gaps in existing suite) or bug report or idea what
could be added or done better.

All contributions should go through our GitHub repository. Bug reports, ideas or
even questions should be raised by opening an issue on the GitHub tracker.
Suggestions for changes in code or documentation should be submitted as a pull
request. However, if you are not sure what to do, feel free to open an issue.
All discussion will then take place on GitHub to keep the development of
pyrosm transparent.

If you decide to contribute to the codebase, ensure that you are using an
up-to-date `master` branch. The latest development version will always be there,
including the documentation (powered by `sphinx`_).


Eight Steps for Contributing
----------------------------

There are seven basic steps to contributing to pyrosm:

1. Fork the pyrosm git repository
2. Create a development environment
3. Install pyrosm dependencies
4. Make a development build of pyrosm
5. Make changes to code and add tests
6. Update the documentation
7. Format code
8. Submit a Pull Request

Each of the steps is detailed below.

1. Fork the pyrosm git repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Git can be complicated for new users, but you no longer need to use command line
to work with git. If you are not familiar with git, we recommend using tools on
GitHub.org, GitHub Desktop or tools with included git like Atom or PyCharm. However, if you
want to use command line, you can fork pyrosm repository using following::

    git clone git@github.com:your-user-name/pyrosm.git pyrosm-yourname
    cd pyrosm-yourname
    git remote add upstream git://github.com/htenkanen/pyrosm.git

This creates the directory pyrosm-yourname and connects your repository to
the upstream (main project) pyrosm repository.

Then simply create a new branch of master branch.


2. Create a development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A development environment is a virtual space where you can keep an independent
installation of pyrosm. This makes it easy to keep both a stable version of
python in one place you use for work, and a development version (which you may
break while playing with code) in another.

An easy way to create a pyrosm development environment is as follows:

- Install either `Anaconda <http://docs.continuum.io/anaconda/>`_ or
  `miniconda <http://conda.pydata.org/miniconda.html>`_
- Make sure that you have cloned the repository
- ``cd`` to the *pyrosm* source directory

Tell conda to create a new environment, named ``pyrosm_dev``, or any other name you would like
for this environment, by running::

      conda create -n pyrosm_dev

This will create the new environment, and not touch any of your existing environments,
nor any existing python installation.

To work in this environment, Windows users should ``activate`` it as follows::

      activate pyrosm_dev

macOS and Linux users should use::

      conda activate pyrosm_dev

You will then see a confirmation message to indicate you are in the new development environment.

To view your environments::

      conda info -e

To return to you home root environment::

      deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`__.

At this point you can easily do a *development* install, as detailed in the next sections.

3. Installing Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

To run *pyrosm* in an development environment, you must first install
*pyrosm*'s dependencies. We suggest doing so using the following commands
(executed after your development environment has been activated)
to ensure compatibility of all dependencies::

    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    conda install geopandas networkx libpysal tqdm pysal mapclassify pytest

This should install all necessary dependencies including optional.

4. Making a development build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once dependencies are in place, make an in-place build by navigating to the git
clone of the *pyrosm* repository and running::

    python setup.py develop

This will install pyrosm into your environment but allows any further changes
without the need of reinstalling new version.

5. Making changes and writing tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*pyrosm* is serious about testing and strongly encourages contributors to embrace
`test-driven development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_.
This development process "relies on the repetition of a very short development cycle:
first the developer writes an (initially failing) automated test case that defines a desired
improvement or new function, then produces the minimum amount of code to pass that test."
So, before actually writing any code, you should write your tests. Often the test can be
taken from the original GitHub issue. However, it is always worth considering additional
use cases and writing corresponding tests.

*pyrosm* uses the `pytest testing system <http://doc.pytest.org/en/latest/>`_.

Writing tests
~~~~~~~~~~~~~

All tests should go into the ``tests`` directory. This folder contains many
current examples of tests, and we suggest looking to these for inspiration.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

The tests can then be run directly inside your Git clone (without having to
install *pyrosm*) by typing::

    pytest

6. Updating the Documentation and User Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*pyrosm* documentation resides in the `docs` folder. Changes to the docs are
make by modifying the appropriate file within `docs`.
*pyrosm* docs us reStructuredText syntax, `which is explained here <http://www.sphinx-doc.org/en/stable/rest.html#rst-primer>`_
and the docstrings follow the `Numpy Docstring standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Once you have made your changes, you may try if they render correctly by building the docs using sphinx.
To do so, you can navigate to the doc folder and type::

    make html

The resulting html pages will be located in docs/build/html. In case of any errors,
you can try to use make html within a new environment based on the libraries in the requirements.txt in the docs folder.

For minor updates, you can skip whole make html part as reStructuredText syntax is
usually quite straightforward.

Updating User Guide
~~~~~~~~~~~~~~~~~~~

Updating user guide might be slightly more complicated as it
consists of collection of reStructuredText files and Jupyter notebooks.
Changes in reStructuredText are straightforward, changes in notebooks should be done using Jupyter. Make sure that all cells have their correct outputs as notebooks
are not executed by readthedocs.

7. Formatting the code
^^^^^^^^^^^^^^^^^^^^^^

Python (PEP8 / black)
~~~~~~~~~~~~~~~~~~~~~

*pyrosm* follows the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ standard
and uses `Black`_ to ensure a consistent code format throughout the project.

CI will run ``black --check`` and fails if there are files which would be
auto-formatted by ``black``. Therefore, it is helpful before submitting code to
auto-format your code::

    black pyrosm

Additionally, many editors have plugins that will apply ``black`` as you edit files.
If you don't have black, you can install it using pip::

    pip install black

8. Submitting a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you've made changes and pushed them to your forked repository, you then
submit a pull request to have them integrated into the *pyrosm* code base.

You can find a pull request (or PR) tutorial in the `GitHub's Help Docs <https://help.github.com/articles/using-pull-requests/>`_.

References
^^^^^^^^^^

These contribution guidelines are largely based on * `momepy`_ -library.

.. _sphinx: https://www.sphinx-doc.org/

.. _Black: https://black.readthedocs.io/en/stable/

.. _momepy: http://docs.momepy.org/en/stable/
