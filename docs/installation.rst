############
Installation
############

You'll need to install ``tedana`` in your python environment.
If you already have a python environment and you want to add tedana, type:

.. code-block:: bash

  pip install tedana

As long as your python environment contains a compatible version of python, this should work.

Transforming report figures into a reference space requires the optional ANTsPyX dependency:

.. code-block:: bash

  pip install "tedana[transforms]"

This extra is only needed when using the ``--xfms`` and ``--reference`` options.

ANTsPyX currently publishes wheels through Python 3.13, so the transform extra is not
available on Python 3.14.


If you want more control over your environment to ensure that every python dependency
is optimized for tedana, 
you can use a program like `conda`_ to create an environment specifically for ``tedana`` with:

.. code-block:: bash

  conda create -n tedenv python=3.12 pip
  conda activate tedenv
  pip install tedana

The above will create a python environment for tedana called ``tedenv``,
and you can enter that environment with ``conda activate tedenv``.

With either of the above methods, `pip` will install tedana and all its dependencies.
tedana's `dependencies are listed here`_.


Once ``tedana`` is installed, there are two ways to run it.
You can run ``tedana``, ``ica_reclassify``, and ``t2smap`` from the command line interface (CLI).
You can see the options for any of these commands using ``--help`` (i.e. ``tedana --help``).

You can also import these commands to run within python using:

.. code-block:: python

  from tedana.workflows import ica_reclassify_workflow, t2smap_workflow, tedana_workflow

API instructions for running these `commands in python are here`_.

If no error occurs, ``tedana`` has correctly installed in your environment!

If you are having trouble solving errors,
`here are places the tedana community monitors`_ to offer support.


.. _commands in python are here: https://tedana.readthedocs.io/en/stable/api.html#module-tedana.workflows
.. _conda: https://www.anaconda.com/download
.. _dependencies are listed here: https://github.com/ME-ICA/tedana/blob/main/pyproject.toml
.. _here are places the tedana community monitors: https://tedana.readthedocs.io/en/stable/support.html
