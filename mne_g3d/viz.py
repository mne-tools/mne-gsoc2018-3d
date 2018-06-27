import os.path as path

import ipyvolume as ipv
import ipywidgets as widgets
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mne.source_estimate import SourceEstimate
from mne.utils import _check_subject, get_subjects_dir
from mne.viz._3d import _handle_time, _limits_to_control_points
import numpy as np
from pythreejs import (BlendFactors, BlendingMode, Equations, ShaderMaterial,
                       Side)

from ._utils import offset_hemi
from .io import read_brain_mesh, read_morph


def plot_brain_mesh(rh_vertices=None,
                    lh_vertices=None,
                    rh_faces=None,
                    lh_faces=None,
                    rh_color='grey',
                    lh_color='grey',
                    act_data=None,
                    cmap_str='auto',
                    offset=0.0,
                    fig_size=(500, 500),
                    azimuth=90,
                    elevation=90):
    u"""Plot triangular format Freesurfer surface of the brain.

    Parameters
    ----------
    rh_vertices : numpy.array, optional
        Array of right hemisphere vertex (x, y, z) coordinates, of size
        number_of_vertices x 3. Default is None.
    lh_vertices : numpy.array, optional
        Array of left hemisphere vertex (x, y, z) coordinates, of size
        number_of_vertices x 3. Default is None.
    rh_faces : numpy.array, optional
        Array defining right hemisphere mesh triangles, of size
        number_of_faces x 3. Default is None.
    lh_faces : numpy.array, optional
        Array defining mesh triangles, of size number_of_faces x 3.
        Default is None.
    rh_color : str | numpy.array, optional
        Color for each point/vertex/symbol of the right hemisphere,
        can be string format, examples for red:’red’, ‘#f00’, ‘#ff0000’ or
        ‘rgb(1,0,0), or rgb array of shape (N, 3). Default value is 'grey'.
    lh_color : str | numpy.array, optional
        Color for each point/vertex/symbol of the left hemisphere,
        can be string format, examples for red:’red’, ‘#f00’, ‘#ff0000’ or
        ‘rgb(1,0,0), or rgb array of shape (N, 3). Default value is 'grey'.
    act_data : numpy.array, optional
        Activation data for for each hemisphere.
    cmap_str : "hot" | "mne" | auto", optional
        Which color map to use. Default value is "auto".
    offset : float | int | None, optional
        If 0.0, the surface will be offset such that the medial wall is
        aligned with the origin. If not 0.0, an additional offset will
        be used. If None no offset will be applied.
    fig_size : (int, int), optional
        Width and height of the figure. Default is (500, 500).
    azimuth : int, optional
        Angle of rotation about the z-axis (pointing up) in degrees.
        Default is 90.
    elevation : int, optional
        Vertical rotation where 90 means ‘up’, -90 means ‘down’, in degrees.
        Default is 90.

    Returns
    -------
    fig : ipyvolume.Figure
        Ipyvolume object presenting the figure.
    rh_mesh : ipyvolume.Mesh
        Ipyvolume object presenting the built mesh for right hemisphere.
    lh_mesh : ipyvolume.Mesh
        Ipyvolume object presenting the built mesh for right hemisphere.
    """
    rh_mesh = None
    lh_mesh = None

    fig = ipv.figure(width=fig_size[0], height=fig_size[1], lighting=True)

    if act_data is not None:
        ctrl_pts, rgb_cmap, scale_pts, _ =\
            _limits_to_control_points('auto',
                                      act_data,
                                      cmap_str,
                                      transparent=False,
                                      fmt='matplotlib')

        if isinstance(rgb_cmap, str):
            # 'hot' color map
            rgb_cmap = cm.get_cmap(rgb_cmap)
            cmap = rgb_cmap(np.arange(rgb_cmap.N))
            alphas = np.ones(rgb_cmap.N)
            step = scale_pts[-1] / rgb_cmap.N
            # coefficients for linear mapping
            # from [ctrl_pts[0], ctrl_pts[1]) interval into [0, 1]
            k = 1 / (ctrl_pts[1] - ctrl_pts[0])
            b = - ctrl_pts[0] * k

            for i in range(0, rgb_cmap.N):
                curr_pos = i * step

                if (curr_pos < ctrl_pts[0]):
                    alphas[i] = 0
                elif (curr_pos >= ctrl_pts[0]) and (curr_pos < ctrl_pts[1]):
                    alphas[i] = k * curr_pos + b
        else:
            # mne color map
            cmap = rgb_cmap(np.arange(rgb_cmap.N))
            alphas = np.ones(rgb_cmap.N)
            step = (scale_pts[-1] - scale_pts[0]) / rgb_cmap.N
            # coefficients for linear mapping into [0, 1]
            k_pos = 1 / (ctrl_pts[1] - ctrl_pts[0])
            k_neg = -k_pos
            b = - ctrl_pts[0] * k_pos

            for i in range(0, rgb_cmap.N):
                curr_pos = i * step + scale_pts[0]

                if (curr_pos > -ctrl_pts[0]) and (curr_pos < ctrl_pts[0]):
                    alphas[i] = 0
                elif (curr_pos >= ctrl_pts[0]) and (curr_pos < ctrl_pts[1]):
                    alphas[i] = k_pos * curr_pos + b
                elif (curr_pos <= -ctrl_pts[0]) and (curr_pos > -ctrl_pts[1]):
                    alphas[i] = k_neg * curr_pos + b

        np.clip(alphas, 0, 1)
        cmap[:, -1] = alphas
        cmap = ListedColormap(cmap)

        dt_min = act_data.min()
        dt_max = act_data.max()
        # data mapping into [0, 1] interval
        k = 1 / (dt_max - dt_min)
        b = 1 - k * dt_max

        act_data = k * act_data + b
        np.clip(act_data, 0, 1)

        lh_act_data = act_data[:len(lh_vertices)]
        rh_act_data = act_data[len(lh_vertices):]
        lh_act_colors = cmap(lh_act_data)
        rh_act_colors = cmap(rh_act_data)
    else:
        rh_act_data = None
        lh_act_data = None
        cmap = None

    if (rh_vertices is not None) and (rh_faces is not None):
        if offset is not None:
            rh_vertices = offset_hemi(rh_vertices, 'rh', offset)

        rh_mesh, _ = plot_hemisphere_mesh(rh_vertices,
                                          rh_faces,
                                          rh_color,
                                          act_colors=rh_act_colors)

    if (lh_vertices is not None) and (lh_faces is not None):
        if offset is not None:
            lh_vertices = offset_hemi(lh_vertices, 'lh', offset)

        lh_mesh, _ = plot_hemisphere_mesh(lh_vertices,
                                          lh_faces,
                                          lh_color,
                                          act_colors=lh_act_colors)

    ipv.style.box_off()
    ipv.style.axes_off()
    ipv.style.background_color('black')

    ipv.view(azimuth, elevation)
    ipv.squarelim()
    ipv.show()

    return fig, rh_mesh, lh_mesh


def plot_hemisphere_mesh(vertices,
                         faces,
                         color='grey',
                         act_colors=None):
    u"""Plot triangular format Freesurfer surface of the brain hemispheres.

    Parameters
    ----------
    vertices : numpy.array
        Array of vertex (x, y, z) coordinates, of size number_of_vertices x 3.
    faces : numpy.array
        Array defining mesh triangles, of size number_of_faces x 3.
    color : str | numpy.array, optional
        Color for each point/vertex/symbol, can be string format, examples for
        red:’red’, ‘#f00’, ‘#ff0000’ or ‘rgb(1,0,0), or rgb array of
        shape (N, 3). Default value is 'grey'.
    act_colors : numpy.array, optional
        Activation data for the given hemispere represented as colors.
    cmap : matplotlib.ListedColormap, optional
        Color map with alpha-channel.

    Returns
    -------
    mesh_widget : ipyvolume.Mesh
        Ipyvolume object presenting the built mesh.
    mesh_overlay : ipyvolume.Mesh
        Ipyvolume object presenting the transparent overlay with
        activation data, if available.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    mesh_widget = ipv.plot_trisurf(x, y, z, triangles=faces, color=color)

    mesh_overlay = None
    # Add mesh overlay and plot data on top of it
    if act_colors is not None:
        mesh_overlay = ipv.plot_trisurf(x,
                                        y,
                                        z,
                                        triangles=faces,
                                        color=act_colors)

        # Tranparency and alpha blending for the new material of the mesh
        mat = ShaderMaterial()
        mat.alphaTest = 0.1
        mat.blending = BlendingMode.CustomBlending
        mat.blendDst = BlendFactors.OneMinusSrcAlphaFactor
        mat.blendEquation = Equations.AddEquation
        mat.blendSrc = BlendFactors.SrcAlphaFactor
        mat.transparent = True
        mat.side = Side.DoubleSide

        mesh_overlay.material = mat

    return mesh_widget, mesh_overlay


def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
                          colormap='auto', time_label='auto',
                          smoothing_steps=10, transparent=None, alpha=1.0,
                          time_viewer=False, subjects_dir=None, views='lat',
                          colorbar=False, clim='auto', cortex='classic',
                          size=800, background='black', foreground=None,
                          initial_time=None, time_unit='s'):
    u"""Plot SourceEstimates with ipyvolume.

    Parameters
    ----------
    stc : SourceEstimates
        The source estimates to plot.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None stc.subject will be used. If that
        is None, the environment will be used.
    surface : str
        The type of surface (inflated, white etc.).
    hemi : str, 'lh' | 'rh' | 'split' | 'both'
        The hemisphere to display.
    colormap : str | np.ndarray of float, shape(n_colors, 3 | 4)
        Name of colormap to use or a custom look up table. If array, must
        be (n x 3) or (n x 4) array for with RGB or RGBA values between
        0 and 255. Default is 'hot'.
    time_label : str | callable | None
        Format of the time label (a format string, a function that maps
        floating point time values to strings, or None for no label). The
        default is ``time=%0.2f ms``.
    smoothing_steps : int
        The amount of smoothing
    transparent : bool | None
        If True, use a linear transparency between fmin and fmid.
        None will choose automatically based on colormap type. Has no effect
        with mpl backend.
    alpha : float
        Alpha value to apply globally to the overlay. Has no effect with mpl
        backend.
    time_viewer : bool
        Display ipybolume time slider.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    views : str | list
        View to use. It must be one of ["lat", "med", "fos", "cau", "dor",
        "ven", "fro", "par"].
    colorbar : bool
        If True, display colorbar on scene.
    clim : str | dict
        Colorbar properties specification. If 'auto', set clim automatically
        based on data percentiles. If dict, should contain:
            ``kind`` : 'value' | 'percent'
                Flag to specify type of limits.
            ``lims`` : list | np.ndarray | tuple of float, 3 elements
                Note: Only use this if 'colormap' is not 'mne'.
                Left, middle, and right bound for colormap.
        Unlike :meth:`stc.plot <mne.SourceEstimate.plot>`, it cannot use
        ``pos_lims``, as the surface plot must show the magnitude.
    cortex : str, tuple, dict, or None
        Specifies how the cortical surface is rendered. Options:
            1. The name of one of the preset cortex styles:
               ``'classic'`` (default), ``'high_contrast'``,
               ``'low_contrast'``, or ``'bone'``.
            2. A color-like argument to render the cortex as a single
               color, e.g. ``'red'`` or ``(0.1, 0.4, 1.)``. Setting
               this to ``None`` is equivalent to ``(0.5, 0.5, 0.5)``.
            3. The name of a colormap used to render binarized
               curvature values, e.g., ``Grays``.
            4. A list of colors used to render binarized curvature
               values. Only the first and last colors are used. E.g.,
               ['red', 'blue'] or [(1, 0, 0), (0, 0, 1)].
            5. A container with four entries for colormap (string
               specifying the name of a colormap), vmin (float
               specifying the minimum value for the colormap), vmax
               (float specifying the maximum value for the colormap),
               and reverse (bool specifying whether the colormap
               should be reversed. E.g., ``('Greys', -1, 2, False)``.
            6. A dict of keyword arguments that is passed on to the
               call to surface.
    size : float or pair of floats
        The size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
        Has no effect with mpl backend.
    background : matplotlib color
        Color of the background of the display window.
    foreground : matplotlib color
        Color of the foreground of the display window. Has no effect with mpl
        backend.
    initial_time : float | None
        The time to display on the plot initially. ``None`` to display the
        first time sample (default).
    time_unit : 's' | 'ms'
        Whether time is represented in seconds ("s", default) or
        milliseconds ("ms").

    Returns
    -------
    figure : ipyvolume.Figure
        An instance of ipyvolume figure.
    """
    if not isinstance(stc, SourceEstimate):
        raise ValueError('stc has to be a surface source estimate')

    views_dict = {'lat': {'elev': 5, 'azim': 0},
                  'med': {'elev': 5, 'azim': 180},
                  'fos': {'elev': 5, 'azim': 90},
                  'cau': {'elev': 5, 'azim': -90},
                  'dor': {'elev': 90, 'azim': 0},
                  'ven': {'elev': -90, 'azim': 0},
                  'fro': {'elev': 5, 'azim': 110},
                  'par': {'elev': 5, 'azim': -110}}

    if views not in views_dict:
        raise ValueError('Views must be one of ["lat", "med", "fos", "cau", '
                         '"dor", "ven", "fro", "par"]. Got {0}.'.format(views))

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    subject = _check_subject(stc.subject, subject, True)

    if colorbar:
        raise NotImplementedError('colobar=True is not yet supported.')

    if not isinstance(colormap, str):
        raise NotImplementedError('Support for "colomap" of a type other' +
                                  ' than str is not yet implemented.')

    if cortex != 'classic':
        raise NotImplementedError('Options for parameter "cortex" ' +
                                  'is not yet supported.')

    if foreground is not None:
        raise NotImplementedError('"foreground" is not yet supported.')

    if hemi == 'split':
        raise NotImplementedError('hemi="split" is not yet implemented.')

    if hemi not in ['lh', 'rh', 'split', 'both']:
        raise ValueError('hemi has to be either "lh", "rh", "split", '
                         'or "both"')

    scaler = 1000. if time_unit == 'ms' else 1.

    if initial_time is None:
        time_idx = 0
    else:
        time_idx = np.argmin(np.abs(stc.times - initial_time / scaler))

    time_label, times = _handle_time(time_label, time_unit, stc.times)

    stc = stc.morph(subject,
                    grade=None,
                    smooth=smoothing_steps,
                    subjects_dir=subjects_dir,
                    subject_from=subject)

    if isinstance(size, int):
        fig_w = size
        fig_h = size
    else:
        fig_w, fig_h = size

    # convert control points to locations in colormap
    ctrl_pts, lim_cmap, scale_pts, transparent = _limits_to_control_points(
        clim, stc.data[:, 0], colormap, transparent, fmt='matplotlib')

    if hemi in ['both', 'split']:
        hemis = ['lh', 'rh']
    else:
        hemis = [hemi]

    fig = ipv.figure(width=fig_w, height=fig_h, lighting=True)

    hemi_meshes = []

    for hemi in hemis:
        hemi_idx = 0 if hemi == 'lh' else 1

        if time_viewer:
            if hemi_idx == 0:
                data_mat = stc.data[:len(stc.vertices[0]), :]
            else:
                data_mat = stc.data[len(stc.vertices[0]):, :]
            # flatten data matrix
            data = data_mat.ravel()
        else:
            if hemi_idx == 0:
                data = stc.data[:len(stc.vertices[0]), time_idx]
            else:
                data = stc.data[len(stc.vertices[0]):, time_idx]

        vertices = stc.vertices[hemi_idx]

        if len(data) > 0:
            if isinstance(lim_cmap, str):
                # 'hot' color map
                rgb_cmap = cm.get_cmap(lim_cmap)
                cmap = rgb_cmap(np.arange(rgb_cmap.N))
                alphas = np.ones(rgb_cmap.N)
                step = scale_pts[-1] / rgb_cmap.N
                # coefficients for linear mapping
                # from [ctrl_pts[0], ctrl_pts[1]) interval into [0, 1]
                k = 1 / (ctrl_pts[1] - ctrl_pts[0])
                b = - ctrl_pts[0] * k

                for i in range(0, rgb_cmap.N):
                    curr_pos = i * step

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

            dt_min = stc.data.min()
            dt_max = stc.data.max()
            # data mapping into [0, 1] interval
            k = 1 / (dt_max - dt_min)
            b = 1 - k * dt_max

            data = k * data + b
            np.clip(data, 0, 1)

            if time_viewer:
                act_colors = cmap(data).reshape(np.r_[data_mat.shape, 4])
                act_colors = act_colors.transpose(1, 0, 2)
            else:
                act_colors = cmap(data)

            mesh_folder = 'surf/{0}.{1}'.format(hemi, surface)
            morph_folder = 'surf/{0}.curv'.format(hemi)

            mesh_path = path.join(subjects_dir, subject, mesh_folder)
            morph_path = path.join(subjects_dir, subject, morph_folder)

            brain_vertices, brain_faces = read_brain_mesh(mesh_path)
            _, brain_color = read_morph(morph_path)

            if len(vertices) != len(brain_vertices):
                raise ValueError(
                    'Should have the same number of vertices,' +
                    '{0} != {1}'.format(
                        len(vertices),
                        len(brain_vertices)
                    ))

            if (brain_vertices is not None) and (brain_faces is not None):
                brain_vertices = offset_hemi(brain_vertices, hemi)

                _, hemi_mesh = plot_hemisphere_mesh(brain_vertices,
                                                    brain_faces,
                                                    brain_color,
                                                    act_colors=act_colors)

                hemi_meshes.append(hemi_mesh)

    if time_viewer:
        control = ipv.animation_control(hemi_meshes,
                                        sequence_length=len(stc.times),
                                        add=False,
                                        interval=100)

        slider = control.children[1]
        slider.readout = False
        slider.value = time_idx
        if isinstance(time_label, str):
            label = widgets.Label(time_label % times[time_idx])
        elif callable(time_label):
            label = widgets.Label(time_label(times[time_idx]))

        # hadler for changing of selected time moment
        def handler(change):
            if isinstance(time_label, str):
                label.value = time_label % times[int(change.new)]
            elif callable(time_label):
                label.value = time_label(times[int(change.new)])

        slider.observe(handler, names='value')
        control = widgets.HBox((*control.children, label))

        ipv.gcc().children += (control,)

    ipv.style.box_off()
    ipv.style.axes_off()
    ipv.style.background_color(background)
    ipv.view(views_dict[views]['azim'], views_dict[views]['elev'])
    ipv.squarelim()
    ipv.show()

    return fig
