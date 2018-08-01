from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np


def _calculate_lut(lim_cmap, alpha, min, mid, max, center=None):
    u"""Transparent color map calculation.

    Parameters
    ----------
    lim_cmap : str | LinearSegmentedColormap
        Color map obtained from MNE._limits_to_control_points.
    alpha : float
        Alpha value to apply globally to the overlay. Has no effect with mpl
        backend.
    min : float
        min value in colormap.
    mid : float
        intermediate value in colormap.
    max : float
        max value in colormap.
    center : float or None
        if not None, center of a divergent colormap, changes the meaning of
        min, max and mid, see :meth:`scale_data_colormap` for further info.

    Returns
    -------
    cmap : matplotlib.ListedColormap
        Color map with transparency channel.
    """
    ctrl_pts = (min, mid, max)

    if center is None:
        # 'hot' or another linear color map
        scale_pts = ctrl_pts
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
        # 'mne' or another divergent color map
        scale_pts = (-1 * max, center, max)
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
