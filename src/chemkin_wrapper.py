import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import fix_dir

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


def init_conditions(working_dir='./', T=1300, P=4,
                    MF={'POSF10325': .004, 'AR': .996}, mode='UV',
                    TIME=2e-3, DELT=2e-5):
    ''' write initial conditions to file '''
    working_dir = fix_dir(working_dir)
    inputs = ''
    if mode == 'UV':
        inputs += 'CONV\n'
    elif mode == 'HP':
        inputs += 'CONP\n'
    inputs += 'RTOL 1.0E-6\n'
    inputs += 'ATOL 1.0E-12\n'
    inputs += 'PRES '+str(P)+'\n'
    inputs += 'TEMP '+str(T)+'\n'
    inputs += 'TIME '+str(TIME)+'\n'
    inputs += 'DELT '+str(DELT)+'\n'
    for s in MF:
        inputs += 'REAC '+s.upper()+' '+str(MF[s])+'\n'
    inputs += 'END\n'
    with open(working_dir+'senkin.inp', 'w') as f:
        f.write(inputs)


def extract_from_outputs(working_dir='./'):
    working_dir = fix_dir(working_dir)
    with open(working_dir+'senkin.ign', 'r') as f:
        lines = f.read()
        lines = lines.split('Time Integration:')[-1]
        d = pd.read_csv(StringIO(lines), sep='\s+')
        # d['t'] = d['t']/1e3
        # print('extract_from_outputs: '+working_dir+'senkin.ign\t',d.shape)
        return d


def plot_outputs(d, names=[], filepath="", log="", title=""):
    if not names:
        names = [n for n in list(d) if n not in set(["t", "T", "P"])]
    plt.figure()
    plt.plot(d["t"], d[names], '.')
    plt.xlabel("Time [us]")
    plt.ylabel("Mole fractions")
    if log == "xy":
        plt.xscale("log")
        plt.yscale("log")
    elif log == "x":
        plt.xscale("log")
    elif log == "y":
        plt.yscale("log")
    elif log == "":
        pass
    else:
        raise ValueError("Invalid log parameter!")
    plt.legend(labels=names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Mole fractions vs time"+title)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath+".pdf")


def chemkin_wrapper(working_dir='./', T=1225, P=1.7,
                    MF={'POSF10325': .004, 'AR': .996}, mode='HP',
                    TIME=2e-3, DELT=1e-6):
    working_dir = fix_dir(working_dir)
    init_conditions(working_dir, T, P, MF, mode, TIME, DELT)
    cwd = os.getcwd()
    os.system('cd '+working_dir+'; ./chem; cd '+cwd)
    os.system('cd '+working_dir+'; ./igsenp; cd '+cwd)
    return extract_from_outputs(working_dir)


def runtime_chemkin_wrapper(working_dir='./'):
    N = 100
    times = np.zeros(N)
    for i in range(N):
        start_time = time.time()
        chemkin_wrapper()
        end_time = time.time()
        times[i] = end_time-start_time
        print(str(i)+" out of "+str(N)+" done")
    np.savetxt('../debug_figures/hist_chemkin_wrapper.csv',
               times, delimiter=',')
    plt.figure()
    plt.hist(times)
    plt.xlabel('time [s]')
    plt.ylabel('count')
    plt.savefig('../debug_figures/hist_chemkin_wrapper.pdf', bins=15)


if __name__ == '__main__':
    # d = chemkin_wrapper()
    # print(d.shape)
    sim = extract_from_outputs("./3/")
    print(sim.head())
