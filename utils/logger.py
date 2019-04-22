import os
import sys

class Logger():

    logfile = None

    def __init__(self, path, name, output = True):
        self.logfile = open(os.path.join(path, name), 'a')
        self.output = output

    def append_line(self, line):
        if self.output:
            print(line)
        self.logfile.write(line + '\n')
        