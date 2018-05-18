from nibabel import freesurfer
import numpy as np


def read_brain_mesh(surface_path):
    u"""Read triangular format Freesurfer brain surface.

    Parameters
    ----------
    surface_path: str
        Path to the brain surface.

    Returns
    -------
    vertices : numpy.array
        Array of vertex (x, y, z) coordinates, of size number_of_vertices x 3.
    faces : numpy.array
        Array defining mesh triangles, of size number_of_faces x 3.
    """
    vertices, faces = freesurfer.read_geometry(surface_path)
    faces = faces.astype(np.uint32)

    return vertices, faces


def read_morph(morph_path):
    u"""Read brain hemisphere morphometry data from Freesurfer 'curv' file.

    Parameters
    ----------
    morph_path : str
        Path to the mophomerty file.

    Returns
    -------
    morph_data : numpy.array
        Original morphometry data.
    color : numpy.array
        Array of colors for each of the vertices.
    """
    morph_data = freesurfer.read_morph_data(morph_path)

    # morphometry (curvature) normalization in order to get gray cortex
    color = (morph_data > 0).astype(float)
    color = (color - 0.5) / 3 + 0.5
    color = color[:, np.newaxis] * [1, 1, 1]

    return morph_data, color
