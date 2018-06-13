import ipyvolume as ipv
import numpy as np
from pythreejs import (BlendFactors, BlendingMode, Equations, ShaderMaterial,
                       Side)

from ._utils import get_mesh_cmap, offset_hemi


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
    act_data : {"lh": numpy.array, "rh": numpy.array}, optional
        Dictionary with activation data for each hemisphere.
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
        cmap = get_mesh_cmap(act_data, cmap_str=cmap_str)
        # data mapping into [0, 1] interval
        arr_act_data = np.concatenate(tuple(act_data.values()))
        dt_min = arr_act_data.min()
        dt_max = arr_act_data.max()

        out_min = 0
        out_max = 1

        k = (out_max - out_min) / (dt_max - dt_min)
        b = out_max - k * dt_max

        rh_act_data = k * act_data.get('rh') + b
        lh_act_data = k * act_data.get('lh') + b

        rh_act_data[rh_act_data < 0] = 0
        lh_act_data[lh_act_data < 0] = 0

        rh_act_data[rh_act_data > 1] = 1.0
        lh_act_data[lh_act_data > 1] = 1.0
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
                                          act_data=rh_act_data,
                                          cmap=cmap)

    if (lh_vertices is not None) and (lh_faces is not None):
        if offset is not None:
            lh_vertices = offset_hemi(lh_vertices, 'lh', offset)

        lh_mesh, _ = plot_hemisphere_mesh(lh_vertices,
                                          lh_faces,
                                          lh_color,
                                          act_data=lh_act_data,
                                          cmap=cmap)

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

        # Tranparency and color blending for the new material of the mesh
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
