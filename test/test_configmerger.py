import sys
import os
from optparse import OptionParser
from config import Config, ConfigMerger
def get_git_home():
    testpath = '.'
    while not '.git' in os.listdir(testpath) and not os.path.abspath(testpath) == '/':
        testpath = os.path.sep.join(('..', testpath))
    if not os.path.abspath(testpath) == '/':
        return os.path.abspath(testpath)
    else:
        raise ValueError, "Not in git repository"
HOME = get_git_home()
sys.path.append(HOME)
from hashmapd.load_config import ConfigLoader

#test, doing config merge directly
def mergeByOverwrite(cfg1, cfg2, key):
    if key == 'train': # need to list any top level keys here
        return "merge"; 
    else:
        return "overwrite";

def test_config():
    cfg = Config(file('test_base.cfg'))
    cfg2 = Config(file('test_override.cfg'))
    merger = ConfigMerger(mergeByOverwrite)
    merger.merge(cfg, cfg2)
    assert cfg.train.mid_layer_sizes[0]==99,"midlayer_sizes[0] should take value from override (99)"
    assert cfg.train.inner_code_length==300 ,"inner_code_length should be as defined in base (300)"
    assert cfg.val1==20 ,"val1 should be as defined in override (10)"
    assert cfg.train.n_ins==20 ,"n_ins should very cleverly combine the formula from base with value from override to give 20"


def test_config_loader():
    cfg = ConfigLoader('./test_').load('test_override')
    assert cfg.train.mid_layer_sizes[0]==99,"midlayer_sizes[0] should take value from override (99)"
    assert cfg.train.inner_code_length==300 ,"inner_code_length should be as defined in base (300)"
    assert cfg.val1==20 ,"val1 should be as defined in override (10)"
    assert cfg.train.n_ins==20 ,"n_ins should very cleverly combine the formula from base with value from override to give 20"
    assert "input_file" not in cfg.keys(), "because 'input' is not in the merge list it should not have values unless explicitly defined in the override"

def test_default():
    cfg = ConfigLoader('./test_').load_default()
    assert cfg.train.inner_code_length==300 ,"inner_code_length should be as defined in base (300)"

