import sys
from optparse import OptionParser
from config import Config, ConfigMerger
try:
  from hashmapd.ConfigLoader import ConfigLoader
except ImportError:
  if 'hashmapd' in sys.path: raise
  sys.path.append('/Users/utunga/Documents/hashmapd/')
  from hashmapd.ConfigLoader import ConfigLoader


#test, doing config merge directly
def mergeByOverwrite(cfg1, cfg2, key):
    if key == 'weights': # need to list any top level keys here
        return "merge"; 
    else:
        return "overwrite";

cfg = Config(file('test_base.cfg'))
cfg2 = Config(file('test_override.cfg'))
merger = ConfigMerger(mergeByOverwrite)
merger.merge(cfg, cfg2)

assert cfg.weights.mid_layer_sizes[0]==99,"midlayer_sizes[0] should take value from override (99)"
assert cfg.weights.inner_code_length==300 ,"inner_code_length should be as defined in base (300)"
assert cfg.val1==20 ,"val1 should be as defined in override (10)"
assert cfg.weights.n_ins==20 ,"n_ins should very cleverly combine the formula from base with value from override to give 20"


#test ConfigLoader approach

cfg = ConfigLoader('./test_').load('override')

assert cfg.weights.mid_layer_sizes[0]==99,"midlayer_sizes[0] should take value from override (99)"
assert cfg.weights.inner_code_length==300 ,"inner_code_length should be as defined in base (300)"
assert cfg.val1==20 ,"val1 should be as defined in override (10)"
assert cfg.weights.n_ins==20 ,"n_ins should very cleverly combine the formula from base with value from override to give 20"



print "Config tests passed"
