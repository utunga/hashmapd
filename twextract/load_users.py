import sys, os, time, csv

filename = sys.argv[1]

print filename

reader = csv.reader(open(filename,"rb"),delimiter=',',quotechar='\"')
for r in reader:
    twname = r[0]
    os.system('PYTHONPATH=`pwd` python twextract/request_queue.py %s -c config/ceres' % twname)
    time.sleep(1)
    

