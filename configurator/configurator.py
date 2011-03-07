import os
from config import Config, ConfigMerger

BASE_DIR = os.path.join('.', 'config')

class ConfigLoader(object):
    """Loads configs in a clever way"""

    def __init__(self, base_dir = BASE_DIR):
        self.base_dir = base_dir
        self.sections_to_merge = ()

    def merge_by_overwrite(self, cfg1, cfg2, key):
        if key in self.sections_to_merge: 
            print key
            return "merge"
        else:
            return "overwrite"
    
    def load(self, config_name):
        base_file = open(os.path.join(self.base_dir, 'base.cfg'))
        base_cfg = Config(base_file)
        if config_name != "base":
            config_name = os.path.splitext(config_name)[0] + '.cfg'
            cfg_file = None
            dirs = ('.', os.path.join('.', 'config'), self.base_dir)
            for directory in dirs:
                try:
                    cfg_file = open(os.path.join(directory, config_name))
                except IOError:
                    pass
            if not cfg_file:
                msg = "Unable to find the config file " + config_name
                raise ValueError, msg
            this_cfg = Config(cfg_file)
            self.sections_to_merge = base_cfg.sections_to_merge 
            merger = ConfigMerger(self.merge_by_overwrite)
            merger.merge(base_cfg, this_cfg)
            return base_cfg, this_cfg
        return base_cfg

    def load_default(self):
        return Config(open(os.path.join(self.base_dir, 'base.cfg')))

def default_config():
    return ConfigLoader().load_default()
    
def load_config(config_name):
    return ConfigLoader().load(config_name)
