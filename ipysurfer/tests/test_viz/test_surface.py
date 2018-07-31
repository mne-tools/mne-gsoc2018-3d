import os

import pytest

from ipysurfer.viz import Surface


@pytest.fixture
def surface(get_subject_info):
    hemi = 'lh'
    surf = 'inflated'
    subject_id = get_subject_info.subject_id
    subjects_dir = get_subject_info.subjects_dir

    return Surface(subject_id, hemi, surf, subjects_dir=subjects_dir)


def test_defaults(get_subject_info):
    hemi = 'lh'
    surf = 'inflated'
    subject_id = get_subject_info.subject_id
    subjects_dir = get_subject_info.subjects_dir
    os.environ['SUBJECTS_DIR'] = subjects_dir

    s = Surface(subject_id, hemi, surf)

    assert s.subject_id == subject_id
    assert s.hemi == hemi
    assert s.surf == surf
    assert s.data_path == os.path.join(subjects_dir, subject_id)
    assert s.units == 'mm'


def test_load_geometry(surface):
    surface.load_geometry()
    # should load mesh geometry
    assert surface.coords is not None
    assert surface.faces is not None
    assert surface.nn is not None


def test_load_curvature(surface):
    surface.load_curvature()
    # should load curvature/morphometry data
    assert surface.curv is not None
    assert surface.bin_curv is not None
    assert surface.grey_curv is not None
