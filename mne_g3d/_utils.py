from matplotlib import cm
from matplotlib.colors import ListedColormap
from mne.viz._3d import _limits_to_control_points
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


def get_mesh_cmap(act_data, cmap_str='auto'):
    u"""Obtain color map with alpha-channel depending on the activation data.

    If input data has negative values, mne color map will be used.
    In another case, 'hot' color map with alpha channel will be used.

    Parameters
    ----------
    act_data : numpy.array | {"lh": numpy.array, "rh": numpy.array}
        Activation data either for a hemispere or for both left and right
        hemispheres.
    cmap_str : "hot" | "mne" | auto", optional
        Which color map to use. Default value is "auto".

    Returns
    -------
    out_cmap: matplotlib.ListedColormap
        Color map with alpha-channel.
    """
    clim = 'auto'

    if isinstance(act_data, dict):
        data = np.concatenate(tuple(act_data.values()))
    else:
        data = act_data.copy()

    ctrl_pts, cmap, scale_pts, _ = _limits_to_control_points(clim,
                                                             data,
                                                             cmap_str,
                                                             transparent=False,
                                                             fmt='matplotlib')

    if isinstance(cmap, str):
        # 'hot' color map
        cmap = cm.get_cmap(cmap)
        out_cmap = cmap(np.arange(cmap.N))
        alphas = np.ones(cmap.N)
        step = scale_pts[-1] / cmap.N
        # coefficients for linear mapping
        # from [ctrl_pts[0], ctrl_pts[1]) interval into [0, 1]
        k = 1 / (ctrl_pts[1] - ctrl_pts[0])
        b = - ctrl_pts[0] * k

        for i in range(0, cmap.N):
            curr_pos = i * step

            if (curr_pos < ctrl_pts[0]):
                alphas[i] = 0
            elif (curr_pos >= ctrl_pts[0]) and (curr_pos < ctrl_pts[1]):
                alphas[i] = k * curr_pos + b
    else:
        # mne color map
        out_cmap = cmap(np.arange(cmap.N))
        alphas = np.ones(cmap.N)
        step = (scale_pts[-1] - scale_pts[0]) / cmap.N

        for i in range(0, cmap.N):
            curr_pos = i * step + scale_pts[0]

            if (curr_pos > -ctrl_pts[0]) and (curr_pos < ctrl_pts[0]):
                alphas[i] = 0

    alphas[alphas > 1] = 1.0
    out_cmap[:, -1] = alphas
    out_cmap = ListedColormap(out_cmap)

    return out_cmap
