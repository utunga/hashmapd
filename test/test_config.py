"""
Tests the base use of the configurator module
"""
import os
from config import Config, ConfigMerger
path = os.path.dirname(os.path.abspath(__file__))

def trivial_resolver(cfg1, cfg2, key):
    return "merge"

def key_resolver(cfg1, cfg2, key):
    if key == "merge":
        return "merge"
    else:
        return "overwrite"

def test_simple():
    test = Config(os.path.join(path, 'test.cfg'))
    base = Config(os.path.join(path, 'base.cfg'))
    assert test.overwrite.x.b == 2
    assert base.overwrite.x.a == 1
           
def test_merge_key():
    test = Config(os.path.join(path, 'test.cfg'))
    base = Config(os.path.join(path, 'base.cfg'))
    assert not hasattr(base.merge, 'test')
    assert base.merge.base == 'from base'
    
    merger = ConfigMerger(key_resolver)
    merger.merge(base, test)
    assert base.merge.base == 'from test'
    assert base.merge.test == 'from test'

def test_merge_overwrite():
    test = Config(os.path.join(path, 'test.cfg'))
    base = Config(os.path.join(path, 'base.cfg'))
    merger = ConfigMerger(key_resolver)
    merger.merge(base, test)
    assert base.baseonly.x.a == 1
    assert base.overwrite.x.a == 2
