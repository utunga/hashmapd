import os, sys, getopt
from hashmapd.ConfigLoader import DefaultConfig, LoadConfig
from hashmapd.render import Render


def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])

    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])

    coords_file = cfg.output.coords_file
    density_plot_file = cfg.output.density_plot_file
    if cfg.input.render_data_has_labels: 
        labels_file = cfg.output.labels_file
    else:
        labels_file = None

    render = Render(coords_file, labels_file)
    render.plot_density(density_plot_file)

if __name__ == '__main__':
    sys.exit(main())
    
    
