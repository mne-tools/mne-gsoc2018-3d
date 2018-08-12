from collections import namedtuple
import os.path as path

import mne
import pytest


SubjectInfo = namedtuple('SubjectInfo', 'subject_id subjects_dir')


@pytest.fixture(scope='session')
def get_subject_info():
    data_path = mne.datasets.sample.data_path()
    subjects_dir = path.join(data_path, 'subjects')

    return SubjectInfo(subject_id='sample', subjects_dir=subjects_dir)
