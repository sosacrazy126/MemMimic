import pytest
from memmimic.api import create_memmimic


def test_memmimic_creation():
    mm = create_memmimic(':memory:')
    assert mm is not None
    
    
def test_memmimic_status():
    mm = create_memmimic(':memory:')
    status = mm.status()
    assert 'status' in status
    assert status['status'] == 'operational'