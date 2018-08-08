from mne.source_estimate import SourceEstimate
from mne.utils import _check_subject
from mne.viz._3d import _handle_time, _limits_to_control_points

from .utils import _get_subjects_dir
from .viz import Brain, TimeViewer


def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
                          colormap='auto', time_label='auto',
                          smoothing_steps=10, transparent=None, alpha=1.0,
                          time_viewer=False, subjects_dir=None, figure=None,
                          views='lat', colorbar=False, clim='auto',
                          cortex='classic', size=800, background='black',
                          foreground=None, initial_time=None, time_unit='s'):
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
        Only str type is supported.
        Name of color map to use or a custom look up table. If array, must
        be (n x 3) or (n x 4) array for with RGB or RGBA values between
        0 and 255. Default is 'hot'. If equals to 'auto', either 'mne' or
        'hot' color map will be selected depending on the input data.
    time_label : str | callable | None
        Format of the time label (a format string, a function that maps
        floating point time values to strings, or None for no label). The
        default is ``time=%0.2f ms``.
    smoothing_steps : int
        The amount of smoothing
    transparent : bool | None
        If True, use linear transparency between fmin and fmid control
        points of a color map. Not yet supported.
    alpha : float
        Alpha value to apply globally to the overlay. Has no effect with mpl
        backend.
    time_viewer : bool
        Display time slider.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    figure : ipyvolume.Figure | list | int | None
        Not yet supported.
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the figure
        by its id or create a new figure with the given id.
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
        Not yet supported.
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
    figure : [ipyvolume.Figure]
        A list of figures.
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

    if (not isinstance(views, list)) and (not isinstance(views, tuple)):
        views = (views,)

    for view in views:
        if view not in views_dict:
            raise ValueError('Views must be one of ["lat", "med", "fos", ' +
                             '"cau", "dor", "ven", "fro", "par"].' +
                             'Got {0}.'.format(views))

    subjects_dir = _get_subjects_dir(subjects_dir=subjects_dir,
                                     raise_error=True)
    subject = _check_subject(stc.subject, subject, True)

    if not isinstance(colormap, str):
        raise ValueError('Support for "colomap" of a type other' +
                         ' than str is not yet implemented.')

    if cortex != 'classic':
        raise ValueError('Options for parameter "cortex" ' +
                         'is not yet supported.')
    if figure is not None:
        raise ValueError('"figure" is not yet supported.')

    if foreground is None:
        foreground = 'black'

    if hemi not in ['lh', 'rh', 'split', 'both']:
        raise ValueError('hemi has to be either "lh", "rh", "split", ' +
                         'or "both"')

    scaler = 1000. if time_unit == 'ms' else 1.
    initial_time /= scaler

    time_label, times = _handle_time(time_label, time_unit, stc.times)

    # convert control points to locations in colormap
    ctrl_pts, lim_cmap, scale_pts, transparent = _limits_to_control_points(
        clim, stc.data.ravel(), colormap, transparent, fmt='matplotlib')

    if hemi in ('both', 'split'):
        hemis = ('lh', 'rh')
    else:
        hemis = (hemi, )

    title = subject if len(hemis) > 1 else '%s - %s' % (subject, hemis[0])
    title = title.capitalize()

    brain_plot = Brain(subject, hemi, surface, size=size,
                       subjects_dir=subjects_dir, title=title,
                       foreground=foreground, views=views)
    for h in hemis:
        if h == 'lh':
            data = stc.data[:len(stc.vertices[0]), :]
            hemi_idx = 0
        else:
            data = stc.data[len(stc.vertices[0]):, :]
            hemi_idx = 1

        vertices = stc.vertices[hemi_idx]

        if len(data) > 0:
            # 'hot' or 'mne' color map
            center = None if isinstance(lim_cmap, str) else 0
            brain_plot.add_data(data, fmin=ctrl_pts[0], fmid=ctrl_pts[1],
                                hemi=h, fmax=ctrl_pts[2], center=center,
                                colormap=lim_cmap, alpha=alpha,
                                initial_time=initial_time,
                                time=times, time_label=time_label,
                                vertices=vertices, colorbar=colorbar)
    if time_viewer:
        TimeViewer(brain_plot)

    brain_plot.show()
    return 0
