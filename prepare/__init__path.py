import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
print("append %s into system" % os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')))