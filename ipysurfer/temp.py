class Brain(object):
    """Class for visualizing a brain using multiple views in mlab

    Parameters
    ----------
    subject_id : str
        subject name in Freesurfer subjects dir
    hemi : str
        hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes.
    surf : str
        freesurfer surface mesh name (ie 'white', 'inflated', etc.)
    title : str
        title for the window
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
               specifiying the name of a colormap), vmin (float
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
        views to use
    offset : bool
        If True, aligs origin with medial wall. Useful for viewing inflated
        surface where hemispheres typically overlap (Default: True)
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
                 cortex="classic", alpha=1.0, size=800, background="black",
                 foreground=None, figure=None, subjects_dir=None,
                 views=['lat'], offset=True, show_toolbar=False,
                 offscreen=False, interaction='trackball', units='mm'):

        if not isinstance(interaction, string_types) or \
                interaction not in ('trackball', 'terrain'):
            raise ValueError('interaction must be "trackball" or "terrain", '
                             'got "%s"' % (interaction,))
        self._units = _check_units(units)
        col_dict = dict(lh=1, rh=1, both=1, split=2)
        n_col = col_dict[hemi]
        if hemi not in col_dict.keys():
            raise ValueError('hemi must be one of [%s], not %s'
                             % (', '.join(col_dict.keys()), hemi))
        # Get the subjects directory from parameter or env. var
        subjects_dir = _get_subjects_dir(subjects_dir=subjects_dir)

        self._hemi = hemi
        if title is None:
            title = subject_id
        self.subject_id = subject_id

        if not isinstance(views, list):
            views = [views]
        n_row = len(views)

        # load geometry for one or both hemispheres as necessary
        offset = None if (not offset or hemi != 'both') else 0.0
        self.geo = dict()
        if hemi in ['split', 'both']:
            geo_hemis = ['lh', 'rh']
        elif hemi == 'lh':
            geo_hemis = ['lh']
        elif hemi == 'rh':
            geo_hemis = ['rh']
        else:
            raise ValueError('bad hemi value')
        geo_kwargs, geo_reverse, geo_curv = self._get_geo_params(cortex, alpha)
        for h in geo_hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and (maybe) curvature
            geo.load_geometry()
            if geo_curv:
                geo.load_curvature()
            self.geo[h] = geo

        # deal with making figures
        self._set_window_properties(size, background, foreground)
        del background, foreground
        figures, _v = _make_viewer(figure, n_row, n_col, title,
                                   self._scene_size, offscreen,
                                   interaction)
        self._figures = figures
        self._v = _v
        self._window_backend = 'Mayavi' if self._v is None else 'TraitsUI'
        for ff in self._figures:
            for f in ff:
                if f.scene is not None:
                    f.scene.background = self._bg_color
                    f.scene.foreground = self._fg_color

        # force rendering so scene.lights exists
        _force_render(self._figures)
        self.toggle_toolbars(show_toolbar)
        _force_render(self._figures)
        self._toggle_render(False)

        # fill figures with brains
        kwargs = dict(geo_curv=geo_curv, geo_kwargs=geo_kwargs,
                      geo_reverse=geo_reverse, subjects_dir=subjects_dir,
                      bg_color=self._bg_color, fg_color=self._fg_color)
        brains = []
        brain_matrix = []
        for ri, view in enumerate(views):
            brain_row = []
            for hi, h in enumerate(['lh', 'rh']):
                if not (hemi in ['lh', 'rh'] and h != hemi):
                    ci = hi if hemi == 'split' else 0
                    kwargs['hemi'] = h
                    kwargs['geo'] = self.geo[h]
                    kwargs['figure'] = figures[ri][ci]
                    kwargs['backend'] = self._window_backend
                    brain = _Hemisphere(subject_id, **kwargs)
                    brain.show_view(view)
                    brains += [dict(row=ri, col=ci, brain=brain, hemi=h)]
                    brain_row += [brain]
            brain_matrix += [brain_row]
        self._toggle_render(True)
        self._original_views = views
        self._brain_list = brains
        for brain in self._brain_list:
            brain['brain']._orient_lights()
        self.brains = [b['brain'] for b in brains]
        self.brain_matrix = np.array(brain_matrix)
        self.subjects_dir = subjects_dir
        self.surf = surf
        # Initialize the overlay and label dictionaries
        self.foci_dict = dict()
        self._label_dicts = dict()
        self.overlays_dict = dict()
        self.contour_list = []
        self.morphometry_list = []
        self.annot_list = []
        self._data_dicts = dict(lh=[], rh=[])
        # note that texts gets treated differently
        self.texts_dict = dict()
        self._times = None
        self.n_times = None

    @property
    def data_dict(self):
        """For backwards compatibility"""
        lh_list = self._data_dicts['lh']
        rh_list = self._data_dicts['rh']
        return dict(lh=lh_list[-1] if lh_list else None,
                    rh=rh_list[-1] if rh_list else None)

    @property
    def labels_dict(self):
        """For backwards compatibility"""
        return {key: data['surfaces'] for key, data in
                self._label_dicts.items()}

    ###########################################################################
    # HELPERS

    def _get_geo_params(self, cortex, alpha=1.0):
        """Return keyword arguments and other parameters for surface
        rendering.

        Parameters
        ----------
        cortex : {str, tuple, dict, None}
            Can be set to: (1) the name of one of the preset cortex
            styles ('classic', 'high_contrast', 'low_contrast', or
            'bone'), (2) the name of a colormap, (3) a tuple with
            four entries for (colormap, vmin, vmax, reverse)
            indicating the name of the colormap, the min and max
            values respectively and whether or not the colormap should
            be reversed, (4) a valid color specification (such as a
            3-tuple with RGB values or a valid color name), or (5) a
            dictionary of keyword arguments that is passed on to the
            call to surface. If set to None, color is set to (0.5,
            0.5, 0.5).
        alpha : float in [0, 1]
            Alpha level to control opacity of the cortical surface.

        Returns
        -------
        kwargs : dict
            Dictionary with keyword arguments to be used for surface
            rendering. For colormaps, keys are ['colormap', 'vmin',
            'vmax', 'alpha'] to specify the name, minimum, maximum,
            and alpha transparency of the colormap respectively. For
            colors, keys are ['color', 'alpha'] to specify the name
            and alpha transparency of the color respectively.
        reverse : boolean
            Boolean indicating whether a colormap should be
            reversed. Set to False if a color (rather than a colormap)
            is specified.
        curv : boolean
            Boolean indicating whether curv file is loaded and binary
            curvature is displayed.

        """
        from matplotlib.colors import colorConverter
        colormap_map = dict(classic=(dict(colormap="Greys",
                                          vmin=-1, vmax=2,
                                          opacity=alpha), False, True),
                            high_contrast=(dict(colormap="Greys",
                                                vmin=-.1, vmax=1.3,
                                                opacity=alpha), False, True),
                            low_contrast=(dict(colormap="Greys",
                                               vmin=-5, vmax=5,
                                               opacity=alpha), False, True),
                            bone=(dict(colormap="bone",
                                       vmin=-.2, vmax=2,
                                       opacity=alpha), True, True))
        if isinstance(cortex, dict):
            if 'opacity' not in cortex:
                cortex['opacity'] = alpha
            if 'colormap' in cortex:
                if 'vmin' not in cortex:
                    cortex['vmin'] = -1
                if 'vmax' not in cortex:
                    cortex['vmax'] = 2
            geo_params = cortex, False, True
        elif isinstance(cortex, string_types):
            if cortex in colormap_map:
                geo_params = colormap_map[cortex]
            elif cortex in lut_manager.lut_mode_list():
                geo_params = dict(colormap=cortex, vmin=-1, vmax=2,
                                  opacity=alpha), False, True
            else:
                try:
                    color = colorConverter.to_rgb(cortex)
                    geo_params = dict(color=color, opacity=alpha), False, False
                except ValueError:
                    geo_params = cortex, False, True
        # check for None before checking len:
        elif cortex is None:
            geo_params = dict(color=(0.5, 0.5, 0.5),
                              opacity=alpha), False, False
        # Test for 4-tuple specifying colormap parameters. Need to
        # avoid 4 letter strings and 4-tuples not specifying a
        # colormap name in the first position (color can be specified
        # as RGBA tuple, but the A value will be dropped by to_rgb()):
        elif (len(cortex) == 4) and (isinstance(cortex[0], string_types)):
            geo_params = dict(colormap=cortex[0], vmin=cortex[1],
                              vmax=cortex[2], opacity=alpha), cortex[3], True
        else:
            try:  # check if it's a non-string color specification
                color = colorConverter.to_rgb(cortex)
                geo_params = dict(color=color, opacity=alpha), False, False
            except ValueError:
                try:
                    lut = create_color_lut(cortex)
                    geo_params = dict(colormap="Greys", opacity=alpha,
                                      lut=lut), False, True
                except ValueError:
                    geo_params = cortex, False, True
        return geo_params

    def get_data_properties(self):
        """ Get properties of the data shown

        Returns
        -------
        props : dict
            Dictionary with data properties

            props["fmin"] : minimum colormap
            props["fmid"] : midpoint colormap
            props["fmax"] : maximum colormap
            props["transparent"] : lower part of colormap transparent?
            props["time"] : time points
            props["time_idx"] : current time index
            props["smoothing_steps"] : number of smoothing steps
        """
        props = dict()
        keys = ['fmin', 'fmid', 'fmax', 'transparent', 'time', 'time_idx',
                'smoothing_steps', 'center']
        try:
            if self.data_dict['lh'] is not None:
                hemi = 'lh'
            else:
                hemi = 'rh'
            for key in keys:
                props[key] = self.data_dict[hemi][key]
        except KeyError:
            # The user has not added any data
            for key in keys:
                props[key] = 0
        return props

    @property
    def overlays(self):
        return self._get_one_brain(self.overlays_dict, 'overlays')

    @property
    def foci(self):
        return self._get_one_brain(self.foci_dict, 'foci')

    @property
    def labels(self):
        return self._get_one_brain(self.labels_dict, 'labels')

    @property
    def contour(self):
        return self._get_one_brain(self.contour_list, 'contour')

    @property
    def annot(self):
        return self._get_one_brain(self.annot_list, 'annot')

    @property
    def texts(self):
        self._get_one_brain([[]], 'texts')
        out = dict()
        for key, val in self.texts_dict.iteritems():
            out[key] = val['text']
        return out

    @property
    def data(self):
        self._get_one_brain([[]], 'data')
        if self.data_dict['lh'] is not None:
            data = self.data_dict['lh'].copy()
        else:
            data = self.data_dict['rh'].copy()
        if 'colorbars' in data:
            data['colorbar'] = data['colorbars'][0]
        return data

    def _check_hemi(self, hemi):
        """Check for safe single-hemi input, returns str"""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                raise ValueError('hemi must not be None when both '
                                 'hemispheres are displayed')
            else:
                hemi = self._hemi
        elif hemi not in ['lh', 'rh']:
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' + extra)
        return hemi

    def _check_hemis(self, hemi):
        """Check for safe dual or single-hemi input, returns list"""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                hemi = ['lh', 'rh']
            else:
                hemi = [self._hemi]
        elif hemi not in ['lh', 'rh']:
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' + extra)
        else:
            hemi = [hemi]
        return hemi

    def _read_scalar_data(self, source, hemi, name=None, cast=True):
        """Load in scalar data from an image stored in a file or an array

        Parameters
        ----------
        source : str or numpy array
            path to scalar data file or a numpy array
        name : str or None, optional
            name for the overlay in the internal dictionary
        cast : bool, optional
            either to cast float data into 64bit datatype as a
            workaround. cast=True can fix a rendering problem with
            certain versions of Mayavi

        Returns
        -------
        scalar_data : numpy array
            flat numpy array of scalar data
        name : str
            if no name was provided, deduces the name if filename was given
            as a source
        """
        # If source is a string, try to load a file
        if isinstance(source, string_types):
            if name is None:
                basename = os.path.basename(source)
                if basename.endswith(".gz"):
                    basename = basename[:-3]
                if basename.startswith("%s." % hemi):
                    basename = basename[3:]
                name = os.path.splitext(basename)[0]
            scalar_data = io.read_scalar_data(source)
        else:
            # Can't think of a good way to check that this will work nicely
            scalar_data = source

        if cast:
            if (scalar_data.dtype.char == 'f' and
                    scalar_data.dtype.itemsize < 8):
                scalar_data = scalar_data.astype(np.float)

        return scalar_data, name

    def _get_display_range(self, scalar_data, min, max, sign):
        if scalar_data.min() >= 0:
            sign = "pos"
        elif scalar_data.max() <= 0:
            sign = "neg"

        # Get data with a range that will make sense for automatic thresholding
        if sign == "neg":
            range_data = np.abs(scalar_data[np.where(scalar_data < 0)])
        elif sign == "pos":
            range_data = scalar_data[np.where(scalar_data > 0)]
        else:
            range_data = np.abs(scalar_data)

        # Get a numeric value for the scalar minimum
        if min is None:
            min = "robust_min"
        if min == "robust_min":
            min = np.percentile(range_data, 2)
        elif min == "actual_min":
            min = range_data.min()

        # Get a numeric value for the scalar maximum
        if max is None:
            max = "robust_max"
        if max == "robust_max":
            max = np.percentile(scalar_data, 98)
        elif max == "actual_max":
            max = range_data.max()

        return min, max

    def _iter_time(self, time_idx, interpolation):
        """Iterate through time points, then reset to current time

        Parameters
        ----------
        time_idx : array_like
            Time point indexes through which to iterate.
        interpolation : str
            Interpolation method (``scipy.interpolate.interp1d`` parameter,
            one of 'linear' | 'nearest' | 'zero' | 'slinear' | 'quadratic' |
            'cubic'). Interpolation is only used for non-integer indexes.

        Yields
        ------
        idx : int | float
            Current index.

        Notes
        -----
        Used by movie and image sequence saving functions.
        """
        current_time_idx = self.data_time_index
        for idx in time_idx:
            self.set_data_time_index(idx, interpolation)
            yield idx

        # Restore original time index
        self.set_data_time_index(current_time_idx)

    ###########################################################################
    # ADDING DATA PLOTS
    def add_overlay(self, source, min=2, max="robust_max", sign="abs",
                    name=None, hemi=None):
        """Add an overlay to the overlay dict from a file or array.

        Parameters
        ----------
        source : str or numpy array
            path to the overlay file or numpy array with data
        min : float
            threshold for overlay display
        max : float
            saturation point for overlay display
        sign : {'abs' | 'pos' | 'neg'}
            whether positive, negative, or both values should be displayed
        name : str
            name for the overlay in the internal dictionary
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        """
        hemi = self._check_hemi(hemi)
        # load data here
        scalar_data, name = self._read_scalar_data(source, hemi, name=name)
        min, max = self._get_display_range(scalar_data, min, max, sign)
        if sign not in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
        old = OverlayData(scalar_data, min, max, sign)
        ol = []
        views = self._toggle_render(False)
        for brain in self._brain_list:
            if brain['hemi'] == hemi:
                ol.append(brain['brain'].add_overlay(old))
        if name in self.overlays_dict:
            name = "%s%d" % (name, len(self.overlays_dict) + 1)
        self.overlays_dict[name] = ol
        self._toggle_render(True, views)

    @verbose
    def add_data(self, array, min=None, max=None, thresh=None,
                 colormap="auto", alpha=1,
                 vertices=None, smoothing_steps=20, time=None,
                 time_label="time index=%d", colorbar=True,
                 hemi=None, remove_existing=False, time_label_size=14,
                 initial_time=None, scale_factor=None, vector_alpha=None,
                 mid=None, center=None, transparent=False, verbose=None):
        """Display data from a numpy array on the surface.

        This provides a similar interface to
        :meth:`surfer.Brain.add_overlay`, but it displays
        it with a single colormap. It offers more flexibility over the
        colormap, and provides a way to display four-dimensional data
        (i.e., a timecourse) or five-dimensional data (i.e., a
        vector-valued timecourse).

        .. note:: ``min`` sets the low end of the colormap, and is separate
                  from thresh (this is a different convention from
                  :meth:`surfer.Brain.add_overlay`).

        Parameters
        ----------
        array : numpy array, shape (n_vertices[, 3][, n_times])
            Data array. For the data to be understood as vector-valued
            (3 values per vertex corresponding to X/Y/Z surface RAS),
            then ``array`` must be have all 3 dimensions.
            If vectors with no time dimension are desired, consider using a
            singleton (e.g., ``np.newaxis``) to create a "time" dimension
            and pass ``time_label=None``.
        min : float
            min value in colormap (uses real min if None)
        mid : float
            intermediate value in colormap (middle between min and max if None)
        max : float
            max value in colormap (uses real max if None)
        thresh : None or float
            if not None, values below thresh will not be visible
        center : float or None
            if not None, center of a divergent colormap, changes the meaning of
            min, max and mid, see :meth:`scale_data_colormap` for further info.
        transparent : bool
            if True: use a linear transparency between fmin and fmid and make
            values below fmin fully transparent (symmetrically for divergent
            colormaps)
        colormap : string, list of colors, or array
            name of matplotlib colormap to use, a list of matplotlib colors,
            or a custom look up table (an n x 4 array coded with RBGA values
            between 0 and 255), the default "auto" chooses a default divergent
            colormap, if "center" is given (currently "icefire"), otherwise a
            default sequential colormap (currently "rocket").
        alpha : float in [0, 1]
            alpha level to control opacity of the overlay.
        vertices : numpy array
            vertices for which the data is defined (needed if len(data) < nvtx)
        smoothing_steps : int or None
            number of smoothing steps (smoothing is used if len(data) < nvtx)
            Default : 20
        time : numpy array
            time points in the data array (if data is 2D or 3D)
        time_label : str | callable | None
            format of the time label (a format string, a function that maps
            floating point time values to strings, or None for no label)
        colorbar : bool
            whether to add a colorbar to the figure
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        remove_existing : bool
            Remove surface added by previous "add_data" call. Useful for
            conserving memory when displaying different data in a loop.
        time_label_size : int
            Font size of the time label (default 14)
        initial_time : float | None
            Time initially shown in the plot. ``None`` to use the first time
            sample (default).
        scale_factor : float | None (default)
            The scale factor to use when displaying glyphs for vector-valued
            data.
        vector_alpha : float | None
            alpha level to control opacity of the arrows. Only used for
            vector-valued data. If None (default), ``alpha`` is used.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see surfer.verbose).

        Notes
        -----
        If the data is defined for a subset of vertices (specified
        by the "vertices" parameter), a smoothing method is used to interpolate
        the data onto the high resolution surface. If the data is defined for
        subsampled version of the surface, smoothing_steps can be set to None,
        in which case only as many smoothing steps are applied until the whole
        surface is filled with non-zeros.

        Due to a Mayavi (or VTK) alpha rendering bug, ``vector_alpha`` is
        clamped to be strictly < 1.
        """
        hemi = self._check_hemi(hemi)
        array = np.asarray(array)

        if center is None:
            if min is None:
                min = array.min() if array.size > 0 else 0
            if max is None:
                max = array.max() if array.size > 0 else 1
        else:
            if min is None:
                min = 0
            if max is None:
                max = np.abs(center - array).max() if array.size > 0 else 1
        if mid is None:
            mid = (min + max) / 2.
        _check_limits(min, mid, max, extra='')

        # Create smoothing matrix if necessary
        if len(array) < self.geo[hemi].x.shape[0]:
            if vertices is None:
                raise ValueError("len(data) < nvtx (%s < %s): the vertices "
                                 "parameter must not be None"
                                 % (len(array), self.geo[hemi].x.shape[0]))
            adj_mat = utils.mesh_edges(self.geo[hemi].faces)
            smooth_mat = utils.smoothing_matrix(vertices, adj_mat,
                                                smoothing_steps)
        else:
            smooth_mat = None

        magnitude = None
        magnitude_max = None
        if array.ndim == 3:
            if array.shape[1] != 3:
                raise ValueError('If array has 3 dimensions, array.shape[1] '
                                 'must equal 3, got %s' % (array.shape[1],))
            magnitude = np.linalg.norm(array, axis=1)
            if scale_factor is None:
                distance = np.sum([array[:, dim, :].ptp(axis=0).max() ** 2
                                   for dim in range(3)])
                if distance == 0:
                    scale_factor = 1
                else:
                    scale_factor = (0.4 * distance /
                                    (4 * array.shape[0] ** (0.33)))
            if self._units == 'm':
                scale_factor = scale_factor / 1000.
            magnitude_max = magnitude.max()
        elif array.ndim not in (1, 2):
            raise ValueError('array has must have 1, 2, or 3 dimensions, '
                             'got (%s)' % (array.ndim,))

        # Process colormap argument into a lut
        lut = create_color_lut(colormap, center=center)
        colormap = "Greys"

        # determine unique data layer ID
        data_dicts = self._data_dicts['lh'] + self._data_dicts['rh']
        if data_dicts:
            layer_id = np.max([data['layer_id'] for data in data_dicts]) + 1
        else:
            layer_id = 0

        data = dict(array=array, smoothing_steps=smoothing_steps,
                    fmin=min, fmid=mid, fmax=max, center=center,
                    scale_factor=scale_factor,
                    transparent=False, time=0, time_idx=0,
                    vertices=vertices, smooth_mat=smooth_mat,
                    layer_id=layer_id, magnitude=magnitude)

        # clean up existing data
        if remove_existing:
            self.remove_data(hemi)

        # Create time array and add label if > 1D
        if array.ndim <= 1:
            initial_time_index = None
        else:
            # check time array
            if time is None:
                time = np.arange(array.shape[-1])
            else:
                time = np.asarray(time)
                if time.shape != (array.shape[-1],):
                    raise ValueError('time has shape %s, but need shape %s '
                                     '(array.shape[-1])' %
                                     (time.shape, (array.shape[-1],)))

            if self.n_times is None:
                self.n_times = len(time)
                self._times = time
            elif len(time) != self.n_times:
                raise ValueError("New n_times is different from previous "
                                 "n_times")
            elif not np.array_equal(time, self._times):
                raise ValueError("Not all time values are consistent with "
                                 "previously set times.")

            # initial time
            if initial_time is None:
                initial_time_index = None
            else:
                initial_time_index = self.index_for_time(initial_time)

            # time label
            if isinstance(time_label, string_types):
                time_label_fmt = time_label

                def time_label(x):
                    return time_label_fmt % x
            data["time_label"] = time_label
            data["time"] = time
            data["time_idx"] = 0
            y_txt = 0.05 + 0.05 * bool(colorbar)

        surfs = []
        bars = []
        glyphs = []
        views = self._toggle_render(False)
        vector_alpha = alpha if vector_alpha is None else vector_alpha
        for brain in self._brain_list:
            if brain['hemi'] == hemi:
                s, ct, bar, gl = brain['brain'].add_data(
                    array, min, mid, max, thresh, lut, colormap, alpha,
                    colorbar, layer_id, smooth_mat, magnitude, magnitude_max,
                    scale_factor, vertices, vector_alpha)
                surfs.append(s)
                bars.append(bar)
                glyphs.append(gl)
                if array.ndim >= 2 and time_label is not None:
                    self.add_text(0.95, y_txt, time_label(time[0]),
                                  name="time_label", row=brain['row'],
                                  col=brain['col'], font_size=time_label_size,
                                  justification='right')
        data['surfaces'] = surfs
        data['colorbars'] = bars
        data['orig_ctable'] = ct
        data['glyphs'] = glyphs

        self._data_dicts[hemi].append(data)

        self.scale_data_colormap(min, mid, max, transparent, center, alpha)

        if initial_time_index is not None:
            self.set_data_time_index(initial_time_index)
        self._toggle_render(True, views)

    def _to_borders(self, label, hemi, borders, restrict_idx=None):
        """Helper to potentially convert a label/parc to borders"""
        if not isinstance(borders, (bool, int)) or borders < 0:
            raise ValueError('borders must be a bool or positive integer')
        if borders:
            n_vertices = label.size
            edges = utils.mesh_edges(self.geo[hemi].faces)
            border_edges = label[edges.row] != label[edges.col]
            show = np.zeros(n_vertices, dtype=np.int)
            keep_idx = np.unique(edges.row[border_edges])
            if isinstance(borders, int):
                for _ in range(borders):
                    keep_idx = np.in1d(self.geo[hemi].faces.ravel(), keep_idx)
                    keep_idx.shape = self.geo[hemi].faces.shape
                    keep_idx = self.geo[hemi].faces[np.any(keep_idx, axis=1)]
                    keep_idx = np.unique(keep_idx)
                if restrict_idx is not None:
                    keep_idx = keep_idx[np.in1d(keep_idx, restrict_idx)]
            show[keep_idx] = 1
            label *= show

    def remove_data(self, hemi=None):
        """Remove data shown with ``Brain.add_data()``.

        Parameters
        ----------
        hemi : str | None
            Hemisphere from which to remove data (default is all shown
            hemispheres).
        """
        hemis = self._check_hemis(hemi)
        for hemi in hemis:
            for brain in self.brains:
                if brain.hemi == hemi:
                    for data in self._data_dicts[hemi]:
                        brain.remove_data(data['layer_id'])
            self._data_dicts[hemi] = []

        # if no data is left, reset time properties
        if all(len(brain.data) == 0 for brain in self.brains):
            self.n_times = self._times = None

    def add_morphometry(self, measure, grayscale=False, hemi=None,
                        remove_existing=True, colormap=None,
                        min=None, max=None, colorbar=True):
        """Add a morphometry overlay to the image.

        Parameters
        ----------
        measure : {'area' | 'curv' | 'jacobian_white' | 'sulc' | 'thickness'}
            which measure to load
        grayscale : bool
            whether to load the overlay with a grayscale colormap
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, data must exist
            for both hemispheres.
        remove_existing : bool
            If True (default), remove old annotations.
        colormap : str
            Mayavi colormap name, or None to use a sensible default.
        min, max : floats
            Endpoints for the colormap; if not provided the robust range
            of the data is used.
        colorbar : bool
            If True, show a colorbar corresponding to the overlay data.

        """
        hemis = self._check_hemis(hemi)
        morph_files = []
        for hemi in hemis:
            # Find the source data
            surf_dir = pjoin(self.subjects_dir, self.subject_id, 'surf')
            morph_file = pjoin(surf_dir, '.'.join([hemi, measure]))
            if not os.path.exists(morph_file):
                raise ValueError(
                    'Could not find %s in subject directory' % morph_file)
            morph_files += [morph_file]

        views = self._toggle_render(False)
        if remove_existing is True:
            # Get rid of any old overlays
            for m in self.morphometry_list:
                if m["colorbar"] is not None:
                    m['colorbar'].visible = False
                m['brain']._remove_scalar_data(m['array_id'])
            self.morphometry_list = []

        for hemi, morph_file in zip(hemis, morph_files):

            if colormap is None:
                # Preset colormaps
                if grayscale:
                    colormap = "gray"
                else:
                    colormap = dict(area="pink",
                                    curv="RdBu",
                                    jacobian_white="pink",
                                    sulc="RdBu",
                                    thickness="pink")[measure]

            # Read in the morphometric data
            morph_data = nib.freesurfer.read_morph_data(morph_file)

            # Get a cortex mask for robust range
            self.geo[hemi].load_label("cortex")
            ctx_idx = self.geo[hemi].labels["cortex"]

            # Get the display range
            min_default, max_default = np.percentile(morph_data[ctx_idx],
                                                     [2, 98])
            if min is None:
                min = min_default
            if max is None:
                max = max_default

            # Use appropriate values for bivariate measures
            if measure in ["curv", "sulc"]:
                lim = np.max([abs(min), abs(max)])
                min, max = -lim, lim

            # Set up the Mayavi pipeline
            morph_data = _prepare_data(morph_data)

            for brain in self.brains:
                if brain.hemi == hemi:
                    self.morphometry_list.append(brain.add_morphometry(
                        morph_data, colormap, measure, min, max, colorbar))
        self._toggle_render(True, views)

    ###########################################################################
    # DATA SCALING / DISPLAY
    def reset_view(self):
        """Orient camera to display original view
        """
        for view, brain in zip(self._original_views, self._brain_list):
            brain['brain'].show_view(view)

    def show_view(self, view=None, roll=None, distance=None, row=-1, col=-1):
        """Orient camera to display view

        Parameters
        ----------
        view : str | dict
            brain surface to view (one of 'lateral', 'medial', 'rostral',
            'caudal', 'dorsal', 'ventral', 'frontal', 'parietal') or kwargs to
            pass to :func:`mayavi.mlab.view()`.
        roll : float
            camera roll
        distance : float | 'auto' | None
            distance from the origin
        row : int
            Row index of which brain to use
        col : int
            Column index of which brain to use

        Returns
        -------
        view : tuple
            tuple returned from mlab.view
        roll : float
            camera roll returned from mlab.roll
        """
        return self.brain_matrix[row][col].show_view(view, roll, distance)

    def set_surf(self, surf):
        """Change the surface geometry

        Parameters
        ----------
        surf : str
            freesurfer surface mesh name (ie 'white', 'inflated', etc.)
        """
        if self.surf == surf:
            return

        views = self._toggle_render(False)

        # load new geometry
        for geo in self.geo.values():
            try:
                geo.surf = surf
                geo.load_geometry()
            except IOError:  # surface file does not exist
                geo.surf = self.surf
                self._toggle_render(True)
                raise

        # update mesh objects (they use a reference to geo.coords)
        for brain in self.brains:
            brain._geo_mesh.data.points = self.geo[brain.hemi].coords
            brain.update_surf()

        self.surf = surf
        self._toggle_render(True, views)

        for brain in self.brains:
            if brain._f.scene is not None:
                brain._f.scene.reset_zoom()

    @verbose
    def scale_data_colormap(self, fmin, fmid, fmax, transparent,
                            center=None, alpha=1.0, verbose=None):
        """Scale the data colormap.

        The colormap may be sequential or divergent. When the colormap is
        divergent indicate this by providing a value for 'center'. The
        meanings of fmin, fmid and fmax are different for sequential and
        divergent colormaps. For sequential colormaps the colormap is
        characterised by::

            [fmin, fmid, fmax]

        where fmin and fmax define the edges of the colormap and fmid will be
        the value mapped to the center of the originally chosen colormap. For
        divergent colormaps the colormap is characterised by::

            [center-fmax, center-fmid, center-fmin, center,
             center+fmin, center+fmid, center+fmax]

        i.e., values between center-fmin and center+fmin will not be shown
        while center-fmid will map to the middle of the first half of the
        original colormap and center-fmid to the middle of the second half.

        Parameters
        ----------
        fmin : float
            minimum value for colormap
        fmid : float
            value corresponding to color midpoint
        fmax : float
            maximum value for colormap
        transparent : boolean
            if True: use a linear transparency between fmin and fmid and make
            values below fmin fully transparent (symmetrically for divergent
            colormaps)
        center : float
            if not None, gives the data value that should be mapped to the
            center of the (divergent) colormap
        alpha : float
            sets the overall opacity of colors, maintains transparent regions
        verbose : bool, str, int, or None
            If not None, override default verbose level (see surfer.verbose).
        """
        divergent = center is not None

        # Get the original colormap
        for h in ['lh', 'rh']:
            data = self.data_dict[h]
            if data is not None:
                table = data["orig_ctable"].copy()
                break

        lut = _scale_mayavi_lut(table, fmin, fmid, fmax, transparent,
                                center, alpha)

        # Get the effective background color as 255-based 4-element array
        geo_actor = self._brain_list[0]['brain']._geo_surf.actor
        if self._brain_list[0]['brain']._using_lut:
            bgcolor = np.mean(
                self._brain_list[0]['brain']._geo_surf.module_manager
                .scalar_lut_manager.lut.table.to_array(), axis=0)
        else:
            bgcolor = geo_actor.property.color
            if len(bgcolor) == 3:
                bgcolor = bgcolor + (1,)
            bgcolor = 255 * np.array(bgcolor)
        bgcolor[-1] *= geo_actor.property.opacity

        views = self._toggle_render(False)
        # Use the new colormap
        for hemi in ['lh', 'rh']:
            data = self.data_dict[hemi]
            if data is not None:
                for surf in data['surfaces']:
                    cmap = surf.module_manager.scalar_lut_manager
                    cmap.load_lut_from_list(lut / 255.)
                    if divergent:
                        cmap.data_range = np.array([center-fmax, center+fmax])
                    else:
                        cmap.data_range = np.array([fmin, fmax])

                    # if there is any transparent color in the lut
                    if np.any(lut[:, -1] < 255):
                        # Update the colorbar to deal with transparency
                        cbar_lut = tvtk.LookupTable()
                        cbar_lut.deep_copy(surf.module_manager
                                           .scalar_lut_manager.lut)
                        alphas = lut[:, -1][:, np.newaxis] / 255.
                        use_lut = lut.copy()
                        use_lut[:, -1] = 255.
                        vals = (use_lut * alphas) + bgcolor * (1 - alphas)
                        cbar_lut.table.from_array(vals)
                        cmap.scalar_bar.lookup_table = cbar_lut
                        cmap.scalar_bar.use_opacity = 1

                # Update the data properties
                data.update(fmin=fmin, fmid=fmid, fmax=fmax, center=center,
                            transparent=transparent)
                # And the hemisphere properties to match
                for glyph in data['glyphs']:
                    if glyph is not None:
                        l_m = glyph.parent.vector_lut_manager
                        l_m.load_lut_from_list(lut / 255.)
                        if divergent:
                            l_m.data_range = np.array(
                                    [center-fmax, center+fmax])
                        else:
                            l_m.data_range = np.array([fmin, fmax])

        self._toggle_render(True, views)

    def set_data_time_index(self, time_idx, interpolation='quadratic'):
        """Set the data time index to show

        Parameters
        ----------
        time_idx : int | float
            Time index. Non-integer values will be displayed using
            interpolation between samples.
        interpolation : str
            Interpolation method (``scipy.interpolate.interp1d`` parameter,
            one of 'linear' | 'nearest' | 'zero' | 'slinear' | 'quadratic' |
            'cubic', default 'quadratic'). Interpolation is only used for
            non-integer indexes.
        """
        from scipy.interpolate import interp1d
        if self.n_times is None:
            raise RuntimeError('cannot set time index with no time data')
        if time_idx < 0 or time_idx >= self.n_times:
            raise ValueError("time index out of range")

        views = self._toggle_render(False)
        for hemi in ['lh', 'rh']:
            for data in self._data_dicts[hemi]:
                if data['array'].ndim == 1:
                    continue  # skip data without time axis

                # interpolation
                if data['array'].ndim == 2:
                    scalar_data = data['array']
                    vectors = None
                else:
                    scalar_data = data['magnitude']
                    vectors = data['array']
                if isinstance(time_idx, float):
                    times = np.arange(self.n_times)
                    scalar_data = interp1d(
                        times, scalar_data, interpolation, axis=1,
                        assume_sorted=True)(time_idx)
                    if vectors is not None:
                        vectors = interp1d(
                            times, vectors, interpolation, axis=2,
                            assume_sorted=True)(time_idx)
                else:
                    scalar_data = scalar_data[:, time_idx]
                    if vectors is not None:
                        vectors = vectors[:, :, time_idx]

                vector_values = scalar_data.copy()
                if data['smooth_mat'] is not None:
                    scalar_data = data['smooth_mat'] * scalar_data
                for brain in self.brains:
                    if brain.hemi == hemi:
                        brain.set_data(data['layer_id'], scalar_data,
                                       vectors, vector_values)
                del brain
                data["time_idx"] = time_idx

                # Update time label
                if data["time_label"]:
                    if isinstance(time_idx, float):
                        ifunc = interp1d(times, data['time'])
                        time = ifunc(time_idx)
                    else:
                        time = data["time"][time_idx]
                    self.update_text(data["time_label"](time), "time_label")

        self._toggle_render(True, views)

    @property
    def data_time_index(self):
        """Retrieve the currently displayed data time index

        Returns
        -------
        time_idx : int
            Current time index.

        Notes
        -----
        Raises a RuntimeError if the Brain instance has not data overlay.
        """
        for hemi in ['lh', 'rh']:
            data = self.data_dict[hemi]
            if data is not None:
                time_idx = data["time_idx"]
                return time_idx
        raise RuntimeError("Brain instance has no data overlay")

    @verbose
    def set_data_smoothing_steps(self, smoothing_steps, verbose=None):
        """Set the number of smoothing steps

        Parameters
        ----------
        smoothing_steps : int
            Number of smoothing steps
        verbose : bool, str, int, or None
            If not None, override default verbose level (see surfer.verbose).
        """
        views = self._toggle_render(False)
        for hemi in ['lh', 'rh']:
            data = self.data_dict[hemi]
            if data is not None:
                adj_mat = utils.mesh_edges(self.geo[hemi].faces)
                smooth_mat = utils.smoothing_matrix(data["vertices"],
                                                    adj_mat, smoothing_steps)
                data["smooth_mat"] = smooth_mat

                # Redraw
                if data["array"].ndim == 1:
                    plot_data = data["array"]
                elif data["array"].ndim == 2:
                    plot_data = data["array"][:, data["time_idx"]]
                else:  # vector-valued
                    plot_data = data["magnitude"][:, data["time_idx"]]

                plot_data = data["smooth_mat"] * plot_data
                for brain in self.brains:
                    if brain.hemi == hemi:
                        brain.set_data(data['layer_id'], plot_data)

                # Update data properties
                data["smoothing_steps"] = smoothing_steps
        self._toggle_render(True, views)

    def index_for_time(self, time, rounding='closest'):
        """Find the data time index closest to a specific time point.

        Parameters
        ----------
        time : scalar
            Time.
        rounding : 'closest' | 'up' | 'down'
            How to round if the exact time point is not an index.

        Returns
        -------
        index : int
            Data time index closest to time.
        """
        if self.n_times is None:
            raise RuntimeError("Brain has no time axis")
        times = self._times

        # Check that time is in range
        tmin = np.min(times)
        tmax = np.max(times)
        max_diff = (tmax - tmin) / (len(times) - 1) / 2
        if time < tmin - max_diff or time > tmax + max_diff:
            err = ("time = %s lies outside of the time axis "
                   "[%s, %s]" % (time, tmin, tmax))
            raise ValueError(err)

        if rounding == 'closest':
            idx = np.argmin(np.abs(times - time))
        elif rounding == 'up':
            idx = np.nonzero(times >= time)[0][0]
        elif rounding == 'down':
            idx = np.nonzero(times <= time)[0][-1]
        else:
            err = "Invalid rounding parameter: %s" % repr(rounding)
            raise ValueError(err)

        return idx

    def set_time(self, time):
        """Set the data time index to the time point closest to time

        Parameters
        ----------
        time : scalar
            Time.
        """
        idx = self.index_for_time(time)
        self.set_data_time_index(idx)

    def _get_colorbars(self, row, col):
        shape = self.brain_matrix.shape
        row = row % shape[0]
        col = col % shape[1]
        ind = np.ravel_multi_index((row, col), self.brain_matrix.shape)
        colorbars = []
        h = self._brain_list[ind]['hemi']
        if self.data_dict[h] is not None and 'colorbars' in self.data_dict[h]:
            colorbars.append(self.data_dict[h]['colorbars'][row])
        if len(self.morphometry_list) > 0:
            colorbars.append(self.morphometry_list[ind]['colorbar'])
        if len(self.contour_list) > 0:
            colorbars.append(self.contour_list[ind]['colorbar'])
        if len(self.overlays_dict) > 0:
            for name, obj in self.overlays_dict.items():
                for bar in ["pos_bar", "neg_bar"]:
                    try:  # deal with positive overlays
                        this_ind = min(len(obj) - 1, ind)
                        colorbars.append(getattr(obj[this_ind], bar))
                    except AttributeError:
                        pass
        return colorbars

    def _colorbar_visibility(self, visible, row, col):
        for cb in self._get_colorbars(row, col):
            if cb is not None:
                cb.visible = visible

    def show_colorbar(self, row=-1, col=-1):
        """Show colorbar(s) for given plot

        Parameters
        ----------
        row : int
            Row index of which brain to use
        col : int
            Column index of which brain to use
        """
        self._colorbar_visibility(True, row, col)

    def hide_colorbar(self, row=-1, col=-1):
        """Hide colorbar(s) for given plot

        Parameters
        ----------
        row : int
            Row index of which brain to use
        col : int
            Column index of which brain to use
        """
        self._colorbar_visibility(False, row, col)

    def close(self):
        """Close all figures and cleanup data structure."""
        for ri, ff in enumerate(self._figures):
            for ci, f in enumerate(ff):
                if f is not None:
                    mlab.close(f)
                    self._figures[ri][ci] = None
        _force_render([])

        # should we tear down other variables?
        if self._v is not None:
            self._v.dispose()
            self._v = None

    def __del__(self):
        if hasattr(self, '_v') and self._v is not None:
            self._v.dispose()
            self._v = None
