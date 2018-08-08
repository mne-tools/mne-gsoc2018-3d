# Ipysurfer

## GSOC 2018: Interactive 3D visualizations of human brain activity in Jupyter Notebook using ipyvolume

This project allows to do interactive 3D visualization of human brain activity inside Jupyter Notebook. It is an alternative to `mayavi` and `PySurfer`-based visualization done in `MNE`, built on top of `ipyvolume` and `ipywidgets`.

## Project information

### Requirements

- [bqplot](https://github.com/bloomberg/bqplot)
- [ipyvolume](https://github.com/maartenbreddels/ipyvolume)
- Matplotlib
- [mne](https://github.com/mne-tools/mne-python)
- [nibabel](https://github.com/nipy/nibabel/)
- NumPy
- [pythreejs](https://github.com/jupyter-widgets/pythreejs)
- SciPy

### Installation

In order to do project installation clone this repository, install all of the required dependencies. Install `ipyvolume` from corresponding GitHub repository, since the changes required by `ipysurfer` have not been released yet. Afterwards, issue a command from top folder of the project:

```bash
pip install -e .
```

## GSOC summary

### Work done

During the program period thses major goas were achieved:

- plotting of cortex meshes in Jupyter Notebook using `ipyvolume`.

- displaying of activity patterns as surface color variations of a mesh.

- creation of a function for plotting `MNE` objects data, instead of raw data. This function can be integrated into `MNE`.

- creation of a stand-alone package that combines code for the above mentioned features, i. e. `ipysurfer`.

- addition of samples of data plotting that uses `ipysurfer` and code documentation.

Also, it some changes were done to `ipyvolume` as part of GSOC:

- addition of alpha blending support to `ipyvolume`, see [#139](https://github.com/maartenbreddels/ipyvolume/pull/139), [#145](https://github.com/maartenbreddels/ipyvolume/pull/145).

- code styling improvements and unit testing configuration changes, see [#146](https://github.com/maartenbreddels/ipyvolume/pull/146), [#142](https://github.com/maartenbreddels/ipyvolume/pull/142).

- a bug fix, see [#158](https://github.com/maartenbreddels/ipyvolume/pull/158).

### What left to do

Several things that should be tackled after this summer of code:

- integration of `ipysurfer` into `MNE`.

- improving package performance by moving some calculations from Python to JavaScript.

- full solution for the issue [#156](https://github.com/maartenbreddels/ipyvolume/issues/156).

### Thanks and credits

I would like to thank my mentors [Jean-Rémi King](https://github.com/kingjr), [Eric Larson](https://github.com/larsoner) and
[Chris Holdgraf](https://github.com/choldgraf) for their help, as well as `ipyvolume` and `bqplot` developers for assisting me in intergration of the corresponding projects to ipysurfer.