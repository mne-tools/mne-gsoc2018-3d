import os

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from ..funcs import _check_units, _get_subjects_dir, _fast_cross_3d


def test_check_units():
    unit_1 = 'm'
    unit_2 = 'mm'
    # should raise a ValueError if unit is not 'm' or 'mm'
    with pytest.raises(ValueError):
        _check_units('cm')
    # should return correct units otherwise
    assert _check_units(unit_1) == unit_1
    assert _check_units(unit_2) == unit_2


def test_get_subjects_dir(get_subject_info):
    subjects_dir = get_subject_info.subjects_dir
    # should raise ValueError for non-existent directory
    with pytest.raises(ValueError):
        _get_subjects_dir(subjects_dir='./no-directory')
    # should raise ValueError if no env variable defined
    with pytest.raises(ValueError):
        _get_subjects_dir()
    # should find a global varible with subjects folder
    os.environ['SUBJECTS_DIR'] = subjects_dir
    assert _get_subjects_dir() == subjects_dir
    # should return correct folder path as output
    assert _get_subjects_dir(subjects_dir) == subjects_dir


def test_fast_cross_3d():
    x = np.ones((600, 3))
    y = -0.5 * np.ones((600, 3))
    assert_array_equal(_fast_cross_3d(x, y), np.cross(x, y))
