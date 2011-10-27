import os, sys, getopt
from hashmapd.load_config import LoadConfig, DefaultConfig
from hashmapd.render import Render

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)
    
    coords_file = cfg.output.coords_file
    density_plot_file = cfg.output.density_plot_file
    labels_file = cfg.output.labels_file

    render = Render(coords_file, labels_file)
    render.max_labels = 1000
    render.plot_density(density_plot_file)
    
