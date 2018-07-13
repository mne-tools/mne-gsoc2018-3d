from nibabel import freesurfer
import numpy as np
from mne import read_source_estimate


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
    color = 0.5 - (color - 0.5) / 3
    color = color[:, np.newaxis] * [1, 1, 1]

    return morph_data, color


def read_activation_data(stc_path, subject_name, subjects_dir=None):
    u"""Read signal activation data.

    Parameters
    ----------
    path : str
        Path to a source-estimate file.
    subject_name : str
        Name of the subject.
    subjects_dir : str
        Path to SUBJECTS_DIR if it is not set in the environment.

    Returns
    -------
    act_data : numpy.array
        Activation data for each hemisphere.
    """
    stc = read_source_estimate(stc_path)
    stc.crop(0.09, 0.09)
    stc = stc.morph(subject_name,
                    grade=None,
                    smooth=10,
                    subjects_dir=subjects_dir,
                    subject_from=subject_name)

    return stc.data[:, 0]
