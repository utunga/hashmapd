"""
Tests the base use of the configurator module
"""
import os
from config import Config, ConfigMerger
path = os.path.dirname(os.path.abspath(__file__))

def trivial_resolver(cfg1, cfg2, key):
    return "merge"

def key_resolver(cfg1, cfg2, key):
    if key == "mergeme":
        return "merge"
    else:
        return "overwrite"

def test_simple():
    test = Config(os.path.join(path, 'test.cfg'))
    base = Config(os.path.join(path, 'base.cfg'))
    assert test.mergeme.test.b == 2
    assert base.mergeme.base.a == 1
   
def test_merge_trivial():
    test = Config(os.path.join(path, 'test.cfg'))
    base = Config(os.path.join(path, 'base.cfg'))
    assert not hasattr(base.mergeme.base, 'b')
    assert not hasattr(base.mergeme, 'test')
    merger = ConfigMerger(trivial_resolver)
    merger.merge(base, test)
    assert base.mergeme.test.b == 2
    assert hasattr(base.mergeme.base, 'a')
    assert hasattr(base.mergeme.base, 'b')
    assert hasattr(base.mergeme, 'test')
    
def test_merge_key():
    test = Config(os.path.join(path, 'test.cfg'))
    base = Config(os.path.join(path, 'base.cfg'))
    assert not hasattr(base.mergeme.base, 'b')
    assert not hasattr(base.mergeme, 'test')
    merger = ConfigMerger(key_resolver)
    merger.merge(base, test)
    assert base.mergeme.test.b == 2
    assert hasattr(base.mergeme.base, 'a')
    assert hasattr(base.mergeme.base, 'b')
    assert hasattr(base.mergeme, 'test')
