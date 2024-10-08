# Spin-based map-making simulation

<p align="center">
  <h1>
  <img src="./maps/scan_fields.png" alt="Logo">
  </h1>
</p>

![build status](https://github.com/yusuke-takase/SBM/actions/workflows/test.yml/badge.svg?branch=master)

This code assesses the impact of differential systematics on CMB polarisation observations.

`ScanFields` such as hit-map and cross-link map, pre-simulated by [Falcons.jl](https://github.com/yusuke-takase/Falcons.jl), are coupled with sky temperature and polarization fields, including systematics effects, at a given $spin$.

# Instllation

```
git clone https://github.com/yusuke-takase/SBM.git
cd SBM
pip install -e .
```

# Database instllation

The SBM needs a database which includes cross-link data in HDF5 format.
By following command, you can install the path of database in local storage.

```
python -m sbm.install_db
```

# Tutorials

There are several tutorials in the [notebooks](https://github.com/yusuke-takase/SBM/tree/master/notebooks).
Although you need to download the required ScanFields datased given by Falcons, you can simulate any time-independent systematics if you have a signal model.
Now, following systematics are implemented:

- Differential gain
- Differential pointing
- Absolute pointing offset with HWP [Y. Takase et al.](https://arxiv.org/abs/2408.03040)
- HWP non-ideality [G. Patanchon et al.](https://iopscience.iop.org/article/10.1088/1475-7516/2024/04/074)
