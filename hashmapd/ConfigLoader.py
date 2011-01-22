from config import Config, ConfigMerger


class ConfigLoader(object):
    """Loads configs in a clever way"""

    def __init__(self, base_path='./config/'):
        self.BASE_PATH = base_path

    def merge_by_overwrite(self, cfg1, cfg2, key):
        # need to list here any top level keys that we want to merge (making this class domain specific unforuntaely, but ah well)
        if key in self.sections_to_merge: #('train', 'tsne'): 
            return "merge"; 
        else:
            return "overwrite";
    
    def load(self, config_name):
        cfg = Config(file(self.BASE_PATH + 'base.cfg'))
        cfg1 = Config(file(self.BASE_PATH + config_name + '.cfg'))
        self.sections_to_merge = cfg.sections_to_merge; 
        merger = ConfigMerger(self.merge_by_overwrite)
        merger.merge(cfg, cfg1)
        return cfg

    def load_default(self):
        return Config(file(self.BASE_PATH + 'base.cfg'))

def DefaultConfig():
    return ConfigLoader().load_default()
    
def LoadConfig(config_name):
    return ConfigLoader().load(config_name);