
import sys
import getopt
import output_from_couch
from hashmapd.ConfigLoader import *

def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])

    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])
    
    output_from_couch.render_token(cfg, 'Yoga', 'out/yoga_density.png')
    
#
#if __name__ == '__main__':
#    sys.exit(main())
#    
#def main():
#    try:
#        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "output="])
#    except getopt.GetoptError, err:
#        # print help information and exit:
#        print str(err) # will print something like "option -a not recognized"
#        usage()
#        sys.exit(2)
#    output = None
#    verbose = False
#    for o, a in opts:
#        if o == "-v":
#            verbose = True
#        elif o in ("-h", "--help"):
#            usage()
#            sys.exit()
#        elif o in ("-o", "--output"):
#            output = a
#        else:
#            assert False, "unhandled option"
    # ...

if __name__ == "__main__":
    main()