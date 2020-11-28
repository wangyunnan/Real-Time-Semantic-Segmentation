from __future__ import print_function
from collections import OrderedDict
import pprint


def CPUinfo():

    procinfo = OrderedDict()
    with open('/proc/cpuinfo') as f:
        for line in f:
            if not line.strip():
                # end of one processor
                CPUinfo = procinfo
                break
            else:
                if len(line.split(':')) == 2:
                    procinfo[line.split(':')[0].strip()] = line.split(':')[1].strip()
                else:
                    procinfo[line.split(':')[0].strip()] = ''
    return CPUinfo


if __name__ == '__main__':
    CPUinfo = CPUinfo()
    print(CPUinfo['model name'].split()[2])