#!/bin/sh

#HTML=all.html
#TXT=all.html
#mv -f $HTML $HTML.bak

for f in $@; do 
    for lang in es pt fr de nl id; do
        curl -s -L "http://$lang.wikipedia.org/wiki/Talk:$f" | html2text  -utf8 -o $lang-$f-talk.txt
        curl -s -L "http://$lang.wikipedia.org/wiki/$f" | html2text  -utf8 -o $lang-$f.txt
    done
done
