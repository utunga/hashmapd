"""
Tests the twextract module
"""
import os
from mock import Mock
#path = os.path.dirname(os.path.abspath(__file__))

def test_simple():
    prod = ProductionClass()
    prod.something = Mock()
    prod.something.method()
    prod.method()
    assert prod.got_called == "unknown"

class ProductionClass(object):
    def method(self):
        self.got_called = "unknown"
        self.something()
    def something(self):
        self.got_called = "production"
        print 'production version of something method got called'
        pass