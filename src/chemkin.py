import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import fix_dir
from src.hychem import CONDITION

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


class CHEMKIN:
    ''' class for chemkin simulation
            excecutes ./chem and ./igsenp
            extracts output from file
            plots mole fraction time history
    '''

    def __init__(self):
        return

    def _init_conditions(self, working_dir='./', cond=CONDITION(),
                         mode='UV', TIME=2e-3, DELT=2e-5):
        ''' (private) write initial conditions to file '''
        working_dir = fix_dir(working_dir)
        T, P, MF = cond.T(), cond.P(), cond.MF()
        inputs = ''
        if mode == 'UV':
            # constant UV
            inputs += 'CONV\n'
        elif mode == 'HP':
            # constant HP
            inputs += 'CONP\n'
        inputs += 'RTOL 1.0E-6\n'
        inputs += 'ATOL 1.0E-12\n'
        inputs += 'PRES '+str(P)+'\n'  # pressure
        inputs += 'TEMP '+str(T)+'\n'  # temperature
        inputs += 'TIME '+str(TIME)+'\n'  # total time of simulation
        inputs += 'DELT '+str(DELT)+'\n'  # time step
        # initial mole fractions
        for s in MF:
            inputs += 'REAC '+s.upper()+' '+str(MF[s])+'\n'
        inputs += 'END\n'
        with open(working_dir+'senkin.inp', 'w') as f:
            f.write(inputs)

    def chemkin_wrapper(self, working_dir='./', cond=CONDITION(),
                        mode='HP', TIME=2e-3, DELT=1e-6):
        ''' (public) wrapper for running chemkin '''
        working_dir = fix_dir(working_dir)
        # write initial condition to file
        self._init_conditions(working_dir, cond, mode, TIME, DELT)
        # change current directory
        cwd = os.getcwd()
        # run ./chem
        os.system('cd '+working_dir+'; ./chem; cd '+cwd)
        # run ./igsenp
        os.system('cd '+working_dir+'; ./igsenp; cd '+cwd)
        # extract mole fraction time history
        self._d = d = self._extract_from_outputs(working_dir)
        return d

    def _extract_from_outputs(self, working_dir='./'):
        ''' (private) extract mole fraction time history from file '''
        working_dir = fix_dir(working_dir)
        with open(working_dir+'senkin.ign', 'r') as f:
            lines = f.read()
            lines = lines.split('Time Integration:')[-1]
            d = pd.read_csv(StringIO(lines), sep='\s+')
            # d['t'] = d['t']/1e3  # time unit
            return d

    def plot_outputs(self, d=None, names=[], filepath="", log="", title=""):
        ''' (public) plot mole fraction time history '''
        # plot the last resul (time history)
        if not d:
            d = self._d
        if not names:
            names = [n for n in list(d) if n not in set(["t", "T", "P"])]
        plt.figure()
        plt.plot(d["t"], d[names], '.')
        plt.xlabel("Time [us]")
        plt.ylabel("Mole fractions")
        # option for log scale
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
