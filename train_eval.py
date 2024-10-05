import os
import sys

"""GPU Number
"""
gpuname = 0

confnames = [
    'real.conf',
    'real_split.conf',
    'real_close.conf',
    'real_close_split.conf'
]
casename = sys.argv[1]
confidx = int(sys.argv[2])
confname = confnames[confidx]
print(confname)
os.system('python -u exp_runner.py --mode train --conf ./confs/{} --case {} --gpu {}'.format(confname, casename, gpuname))
os.system('python -u exp_runner.py --mode intergaze --conf ./confs/{} --case {} --gpu {} --is_continue'.format(confname, casename, gpuname))
