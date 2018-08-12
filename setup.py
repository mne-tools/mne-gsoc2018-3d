#! /usr/bin/env python
from setuptools import setup, find_packages

DISTNAME = 'ipysurfer'
DESCRIPTION = 'Interactive 3D visualizations of human brain activity in the Jupyter Notebook.'
AUTHOR = 'Oleh Kozynets'
URL = 'https://github.com/mne-tools/mne-gsoc2018-3d'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mne-tools/mne-gsoc2018-3d'

EXTERNAL_PACKAGES = ('ipyvolume', 'matplotlib', 'mne', 'numpy', 'nibabel', 'pythreejs',
                     'scipy', 'bqplot')
setup(
    name=DISTNAME,
    author=AUTHOR,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version='0',
    download_url=DOWNLOAD_URL,
    long_description=open('README.md').read(),
    platforms='any',
    packages=find_packages(),
    install_requires=EXTERNAL_PACKAGES)
