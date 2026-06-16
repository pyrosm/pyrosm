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

There are eight basic steps to contributing to pyrosm:

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
    git remote add upstream https://github.com/pyrosm/pyrosm.git

This creates the directory pyrosm-yourname and connects your repository to
the upstream (main project) pyrosm repository.

Then simply create a new branch of master branch.


2. Create a development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A development environment is a virtual space where you can keep an independent
installation of pyrosm. This makes it easy to keep both a stable version of
python in one place you use for work, and a development version (which you may
break while playing with code) in another.

We recommend installing the dependencies from conda-forge with `mamba
<https://mamba.readthedocs.io/>`_ (or its standalone variant `micromamba
<https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_), a fast
drop-in replacement for ``conda``. If you don't have it yet, download and install
mamba via Miniforge from the `conda-forge download page
<https://conda-forge.org/download/>`_ -- it ships mamba preconfigured with the
conda-forge channel.

Make sure you have cloned the repository and ``cd`` into the *pyrosm* source
directory. The ``ci/`` folder ships ready-made environment files, one per
supported Python version (3.10--3.14), that pin every dependency. Create an
environment from one of them (it is named ``test`` by default), e.g. for Python
3.14::

      mamba env create -f ci/314-conda.yaml

or, with micromamba::

      micromamba create -f ci/314-conda.yaml

Then activate it::

      mamba activate test

(with micromamba, use ``micromamba activate test``). You will then see a
confirmation message indicating you are in the development environment. At this
point you can do a *development* install, as detailed in the next sections.

3. Installing Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ci/<version>-conda.yaml`` environment file you used in the previous step
already installs all of *pyrosm*'s dependencies from conda-forge -- the required
runtime packages (``geopandas``, ``protobuf``, ``python-rapidjson``), the build
tools (``cython``, ``cykhash``), the optional graph backends (``networkx``,
``python-igraph``, ``pandarm``) and the tools for running the tests and the
formatter (``pytest``, ``pytest-cov``, ``black``) -- so there is nothing extra to
install. If you ever need to refresh them, re-create the environment from the
same file.

4. Making a development build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*pyrosm* contains Cython (``*.pyx``) sources that must be compiled before use.
Once the environment is in place, make an in-place build by navigating to the git
clone of the *pyrosm* repository and running::

    pip install -e . --no-build-isolation

This installs pyrosm in editable mode and compiles the Cython extensions against
the build dependencies provided by the environment (``cython`` and ``cykhash``),
instead of refetching and recompiling them in an isolated build environment.
Editing any ``.pyx``/``.pxd`` file requires re-running this command to rebuild.

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
