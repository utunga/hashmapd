from config import Config, ConfigMerger


class ConfigLoader(object):
    """Loads configs in a clever way"""

    def __init__(self, base_path='../config/'):
        self.BASE_PATH = base_path
    
    def merge_by_overwrite(self, cfg1, cfg2, key):
        if key == 'weights': # need to list any top level keys here (making this class domain specific, but thats OK)
            return "merge"; 
        else:
            return "overwrite";
    
    def load(self, config_name):
        cfg = Config(file(self.BASE_PATH + 'base.cfg'))
        cfg1 = Config(file(self.BASE_PATH + config_name + '.cfg'))
        merger = ConfigMerger(self.merge_by_overwrite)
        merger.merge(cfg, cfg1)
        return cfg

    def load_default(self):
        return Config(file(self.BASE_PATH + 'base.cfg'))
