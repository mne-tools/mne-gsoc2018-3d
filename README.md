# Ipysurfer

This project allows to do interactive 3D visualization of human brain activity inside Jupyter Notebook. It can be regarded as an alternative to `mayavi` and `PySurfer`-based human brain visualization, built on top of `ipyvolume` and `ipywidgets`.

## Requirements

- [bqplot](https://github.com/bloomberg/bqplot)
- [ipyvolume](https://github.com/maartenbreddels/ipyvolume)
- Matplotlib
- [mne](https://github.com/mne-tools/mne-python)
- [nibabel](https://github.com/nipy/nibabel/)
- NumPy
- [pythreejs](https://github.com/jupyter-widgets/pythreejs)
- SciPy

## Installation

In order to do project installation clone this repository, install all of the required dependencies. Install `ipyvolume` from corresponding GitHub repository, since the changes required by `ipysurfer` have not been released yet. Afterwards, issue a command from top folder of the project:

```bash
pip install -e .
```
