
import ipyvolume as ipv
from pythreejs import (BlendFactors, BlendingMode, Equations, ShaderMaterial,
                       Side)
from ._utils import Surface


lh_viewdict = {'lateral': {'v': (180., 90.), 'r': 90.},
               'medial': {'v': (0., 90.), 'r': -90.},
               'rostral': {'v': (90., 90.), 'r': -180.},
               'caudal': {'v': (270., 90.), 'r': 0.},
               'dorsal': {'v': (180., 0.), 'r': 90.},
               'ventral': {'v': (180., 180.), 'r': 90.},
               'frontal': {'v': (120., 80.), 'r': 106.739},
               'parietal': {'v': (-120., 60.), 'r': 49.106}}
rh_viewdict = {'lateral': {'v': (180., -90.), 'r': -90.},
               'medial': {'v': (0., -90.), 'r': 90.},
               'rostral': {'v': (-90., -90.), 'r': 180.},
               'caudal': {'v': (90., -90.), 'r': 0.},
               'dorsal': {'v': (180., 0.), 'r': 90.},
               'ventral': {'v': (180., 180.), 'r': 90.},
               'frontal': {'v': (60., 80.), 'r': -106.739},
               'parietal': {'v': (-60., 60.), 'r': -49.106}}
viewdicts = dict(lh=lh_viewdict, rh=rh_viewdict)


class Brain:
    u"""Class for visualizing a brain using multiple views in ipyvolume.

    Parameters
    ----------
    subject_id : str
        subject name in Freesurfer subjects dir.
    hemi : str
        hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes.
    surf : str
        freesurfer surface mesh name (ie 'white', 'inflated', etc.).
    title : str
        title for the window.
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
    alpha : float in [0, 1]
        Alpha level to control opacity of the cortical surface.
    size : float or pair of floats
        the size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
    background : matplotlib color
        Color of the background.
    foreground : matplotlib color
        Color of the foreground (will be used for colorbars and text).
        None (default) will use black or white depending on the value
        of ``background``.
    figure : list of mayavi.core.scene.Scene | None | int
        If None (default), a new window will be created with the appropriate
        views. For single view plots, the figure can be specified as int to
        retrieve the corresponding Mayavi window.
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment
        variable.
    views : list | str
        views to use.
    offset : bool
        If True, aligs origin with medial wall. Useful for viewing inflated
        surface where hemispheres typically overlap (Default: True).
    show_toolbar : bool
        If True, toolbars will be shown for each view.
    offscreen : bool
        If True, rendering will be done offscreen (not shown). Useful
        mostly for generating images or screenshots, but can be buggy.
        Use at your own risk.
    interaction : str
        Can be "trackball" (default) or "terrain", i.e. a turntable-style
        camera.
    units : str
        Can be 'm' or 'mm' (default).

    Attributes
    ----------
    annot : list
        List of annotations.
    brains : list
        List of the underlying brain instances.
    contour : list
        List of the contours.
    foci : foci
        The foci.
    labels : dict
        The labels.
    overlays : dict
        The overlays.
    texts : dict
        The text objects.
    """

    def __init__(self, subject_id, hemi, surf, title=None,
                 cortex='classic', alpha=1.0, size=800, background='black',
                 foreground=None, figure=None, subjects_dir=None,
                 views=['lat'], offset=True, show_toolbar=False,
                 offscreen=False, interaction=None, units='mm'):
        # surf =  surface
        # implement title
        if cortex != 'classic':
            raise NotImplementedError('Options for parameter "cortex" ' +
                                      'is not yet supported.')
        if hemi == 'split':
            raise NotImplementedError('hemi="split" is not yet implemented.')

        if figure is not None:
            raise NotImplementedError('figure parameter' +
                                      'has not been implemented yet.')

        if interaction is not None:
            raise NotImplementedError('interaction parameter' +
                                      'has not been implemented yet.')

        self._units = units

        # load geometry for one or both hemispheres as necessary
        offset = None if (not offset or hemi != 'both') else 0.0

        if hemi in ('both', 'split'):
            hemis = ('lh', 'rh')
        elif hemi in ('lh', 'rh'):
            hemis = (hemi, )
        else:
            raise ValueError('hemi has to be either "lh", "rh", "split", ' +
                             'or "both"')

        if isinstance(size, int):
            fig_w = size
            fig_h = size
        else:
            fig_w, fig_h = size

        self.geo = {}
        self._fig = ipv.figure(width=fig_w, height=fig_h, lighting=True)
        self._hemi_meshes = {}

        for h in hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and curvature
            geo.load_geometry()
            geo.load_curvature()

            _, hemi_mesh = _plot_hemisphere_mesh(geo.coords,
                                                 geo.faces,
                                                 geo.grey_curv)
            self.geo[h] = geo
            self._hemi_meshes[hemi] = hemi_mesh

        ipv.style.box_off()
        ipv.style.axes_off()
        ipv.style.background_color(background)
        # TODO: how extract azimuth and elevation from cuurent
        # views representation
        # or should I stick to the ones I have from MNE?
        # ipv.view(views_dict[views]['azim'], views_dict[views]['elev'])
        ipv.squarelim()
        ipv.show()


def _plot_hemisphere_mesh(vertices,
                          faces,
                          color='grey',
                          act_data=None,
                          cmap=None):
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
    act_data : numpy.array, optional
        Activation data for the given hemispere.
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
    if (act_data is not None) and (cmap is not None):
        act_colors = cmap(act_data)

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
