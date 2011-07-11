#!/bin/sh

#HTML=all.html
#TXT=all.html
#mv -f $HTML $HTML.bak

#http://bash.org/?random

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do  
    curl -s "http://bash.org/?random" | html2text  -utf8 -o bash-random-$1$i.txt
done
