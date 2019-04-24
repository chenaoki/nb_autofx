from const import pairs
import os, sys, commands

url = 'http://www.forextester.com/templates/data/files/{0}.zip'

for pair in pairs:
    cmd = 'wget '+url.format(pair)
    print cmd
    commands.getoutput(cmd)
    