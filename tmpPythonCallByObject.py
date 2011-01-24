# just Miles trying to get his head around this really disturbing (to a C#/Java programmer)
# 'call by object' insanity of neither pass by value or pass by reference that is how python 'does things'
# http://effbot.org/zone/call-by-object.htm
from copy import copy
class quick_test(object):
   
    def __init__(self,  mid_layer, last):
     
        tmp = copy(mid_layer)
        self.mid_layer = mid_layer
        self.last = last
        self.crazy()


        self.mid_layer = tmp
        self.last = last
        self.less_crazy()
        
    def crazy(self):
        print self.mid_layer
        #prints [200, 100] as you'd expect

        tmp = self.mid_layer
        tmp.append(self.last)
        print self.mid_layer
        #prints "[200, 100, 10]" wha?
        
        tmp = self.mid_layer
        tmp.append(self.last)
        print self.mid_layer
        #prints "[200, 100, 10, 10]" huh?
        #good god this behaviour is disturbing .. wtf?
        
    def less_crazy(self):
        print self.mid_layer
        #prints "[200, 100]" *(due to use of copy() to stash a copy of list above)
        
        tmp = self.mid_layer + [self.last]
        
        print tmp
        #prints "[200, 100, 10]" 
        
        print self.mid_layer
        #prints "[200, 100]" as *I* would expect, phew *unchanged
        
if __name__ == '__main__':
     ok = quick_test([200,100],10)
     