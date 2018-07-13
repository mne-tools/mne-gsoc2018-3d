from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np


def offset_hemi(vertices, hemi, offset=0.0):
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


def _calculate_cmap(lim_cmap, alpha, ctrl_pts, scale_pts):
    u"""Transparent color map calculation.

    Parameters
    ----------
    lim_cmap : str | LinearSegmentedColormap
        Color map obtained from MNE._limits_to_control_points.
    alpha : float
        Alpha value to apply globally to the overlay. Has no effect with mpl
        backend.
    ctrl_pts : tuple(float)
        Color map control points.
    scale_pts : tuple(float)
        Data scale control points.

    Returns
    -------
    cmap : matplotlib.ListedColormap
        Color map with transparency channel.
    """
    if isinstance(lim_cmap, str):
        # 'hot' color map
        rgb_cmap = cm.get_cmap(lim_cmap)
        # take 60% of hot color map, so it will be consistent
        # with mayavi plots
        cmap_size = int(rgb_cmap.N * 0.6)
        cmap = rgb_cmap(np.arange(rgb_cmap.N))[rgb_cmap.N - cmap_size:, :]
        alphas = np.ones(cmap_size)
        step = 2 * (scale_pts[-1] - scale_pts[0]) / rgb_cmap.N
        # coefficients for linear mapping
        # from [ctrl_pts[0], ctrl_pts[1]) interval into [0, 1]
        k = 1 / (ctrl_pts[1] - ctrl_pts[0])
        b = - ctrl_pts[0] * k

        for i in range(0, cmap_size):
            curr_pos = i * step + scale_pts[0]

            if (curr_pos < ctrl_pts[0]):
                alphas[i] = 0
            elif (curr_pos < ctrl_pts[1]):
                alphas[i] = k * curr_pos + b
    else:
        # mne color map
        rgb_cmap = lim_cmap
        cmap = rgb_cmap(np.arange(rgb_cmap.N))
        alphas = np.ones(rgb_cmap.N)
        step = (scale_pts[-1] - scale_pts[0]) / rgb_cmap.N
        # coefficients for linear mapping into [0, 1]
        k_pos = 1 / (ctrl_pts[1] - ctrl_pts[0])
        k_neg = -k_pos
        b = - ctrl_pts[0] * k_pos

        for i in range(0, rgb_cmap.N):
            curr_pos = i * step + scale_pts[0]

            if -ctrl_pts[0] < curr_pos < ctrl_pts[0]:
                alphas[i] = 0
            elif ctrl_pts[0] <= curr_pos < ctrl_pts[1]:
                alphas[i] = k_pos * curr_pos + b
            elif -ctrl_pts[1] < curr_pos <= -ctrl_pts[0]:
                alphas[i] = k_neg * curr_pos + b

    alphas *= alpha
    np.clip(alphas, 0, 1)
    cmap[:, -1] = alphas
    cmap = ListedColormap(cmap)

    return cmap
