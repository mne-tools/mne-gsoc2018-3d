import numpy as np


def _offset_hemi(vertices, hemi, offset=0.0):
    u"""Offset hemispere.

    Parameters
    ----------
    vertices : numpy.array
        Array of hemisphere vertex (x, y, z) coordinates, of size
        number_of_vertices x 3.
    hemi : {'lh', 'rh'}
        Which hemisphere to offset.
    offset : float | int, optional
        If 0.0, the surface will be offset such that the medial
        wall is aligned with the origin. If not 0.0, an
        additional offset will be used.

    Returns
    -------
    vertices : numpy.array
        Array of offset hemisphere vertex (x, y, z) coordinates, of size
        number_of_vertices x 3.
    """
    hemis = ('lh', 'rh')

    if hemi not in hemis:
        raise ValueError('hemi should be either "lh" or "rh", given value {0}'.
                         format(hemi))

    if (not isinstance(offset, float)) and (not isinstance(offset, int)):
        raise ValueError('offset should either float or int, given type {0}'.
                         format(type(offset).__name__))

    vertices = vertices.copy()

    if hemi == hemis[0]:
        vertices[:, 0] -= (np.max(vertices[:, 0]) + offset)
    else:
        vertices[:, 0] -= (np.min(vertices[:, 0]) + offset)

    return vertices
