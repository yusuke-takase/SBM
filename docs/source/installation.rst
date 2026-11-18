Installation
============

SBM needs Python 3.10 or later, NumPy (1.x or 2.x), healpy, h5py and ducc0.
Install it from the source:

.. code-block:: bash

   git clone https://github.com/yusuke-takase/SBM
   cd SBM
   pip install -e .            # or: uv pip install -e .

Optional dependencies are imported only by the functions that use them:

.. list-table::
   :header-rows: 1

   * - Extra
     - Package
     - Needed by
   * - ``plot``
     - matplotlib
     - :func:`sbm.get_cmap`

For example ``pip install -e ".[plot]"``. The tests need only pytest
(``pip install -e ".[test]"``) and run with ``pytest`` from any directory.
