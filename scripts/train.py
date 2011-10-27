from hashmapd.load_config import LoadConfig
from hashmapd.SMH import train_SMH
from optparse import OptionParser

if __name__ == '__main__':

    #load config
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)

    #print config (for debug/logging purposes)
    print "PARAMETERS"
    for section in [cfg.shape, cfg.train]:
        for (key, value) in section.iteritems():
            print '  {0} = {1}'.format(key, value)

    #actually run the train
    smh = train_SMH('data',
            mid_layer_sizes = list(cfg.shape.mid_layer_sizes), 
            inner_code_length = cfg.shape.inner_code_length, 
            **cfg.train)

        
    
    
