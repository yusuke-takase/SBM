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
cd sbm
pip install -e .
```

# Tutorials

There are several tutorials in the [notebooks](https://github.com/yusuke-takase/SBM/tree/master/notebooks).
Although you need to download the required ScanFields datased given by Falcons, you can simulate any time-independent systematics if you have a signal model.
Now, following systematics are implemented:

- Differential gain
- Differential pointing
