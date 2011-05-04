import os, sys, getopt

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
    if cfg.input.render_data_has_labels: 
        labels_file = cfg.output.labels_file
    else:
        labels_file = None

    render = Render(coords_file, labels_file)
    render.plot_density(density_plot_file)
    
