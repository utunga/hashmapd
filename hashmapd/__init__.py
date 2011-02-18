from hashmapd.ConfigLoader import *
from hashmapd.HiddenLayer import *
from hashmapd.SMH import *
from hashmapd.utils import *
from hashmapd.querycouch import *
#NOTE2ED I'm thoroughly confused by python's package maanagement defaults
#         is this the 'standard' way to do this or am I missing things?
#        at the moment I'm working on the idea that into __init__.py its common to put imports
#       for all the bits we expect callees to want to have 'easy' access to and require import hashmapd.xyz for more internal 'xyz' bits ? - MKT 2010--01

#Hi Miles
#No this is an unusual way of doing things :-) Generally __init__.py is and
#empty, and you should also avoid "from module import *" as it causes
#namespace pollution, that will inevitably lead to hard to find bugs ...
#An opportunity here for me to refactor!
