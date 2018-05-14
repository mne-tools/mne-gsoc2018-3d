import logging

import ipyvolume as ipv
import nibabel
import numpy as np

def read_brain_mesh(surface_path: str):
    """Function reads triangular format Freesurfer brain surface.

    Parameters
    ----------
    surface_path: :obj: `str`
        Path to the brain surface.

    Returns
    -------
    :obj: `numpy.array`
        Array of vertex (x, y, z) coordinates, of size number_of_vertices x 3.
    :obj: `numpy.array`
        Array defining mesh triangles, of size number_of_faces x 3.

    """

    vertices, faces = nibabel.freesurfer.read_geometry(surface_path)
    faces = faces.astype(np.uint32)

    return vertices, faces


def plot_brain_mesh(vertices, faces,  color='grey'):
    """Function for plotting triangular format Freesurfer surface of the brain.

    Parameters
    ----------
    vertices : :obj: `numpy.array`
        Array of vertex (x, y, z) coordinates, of size number_of_vertices x 3.
    faces : :obj: `numpy.array`
        Array defining mesh triangles, of size number_of_faces x 3.
    color : :obj: `str`, :obj: `numpy.array`, optional
        Color for each point/vertex/symbol, can be string format, examples for red:’red’, ‘#f00’, 
        ‘#ff0000’ or ‘rgb(1,0,0), or rgb array of shape (N, 3). Default value is 'grey'.

    Returns
    -------
    :obj: `ipyvolume.Mesh`
        Ipyvolume object presenting the built mesh.

    """

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    mesh_widget = ipv.plot_trisurf(x, y, z, triangles=faces, color=color)

    return mesh_widget
