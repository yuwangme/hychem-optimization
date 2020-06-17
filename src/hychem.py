import numpy as np
from scipy.special import expit
from scipy.interpolate import interp1d
from src.utils import fix_dir

N_C = 11
N_H = 22
FUEL_NAME = 'POSF10325'
# SIX_R = ['H', 'CH3', 'O', 'OH', 'O2', 'HO2']
SIX_R = ['H', 'CH3', 'OH', 'O2', 'HO2', 'O']
# SIX_RH = ['H2', 'CH4', 'OH', 'H2O', 'HO2', 'H2O2']
SIX_RH = ['H2', 'CH4', 'H2O', 'HO2', 'H2O2', 'OH']
DECOMP_PROD = ['C2H4', 'C3H6', 'iC4H8', 'C6H6', 'C6H5CH3', 'H', 'CH3']
H_ABS_PROD = ['CH4']+DECOMP_PROD
H_ABS_PROD_C = [1, 2, 3, 4, 6, 7, 0, 1]
H_ABS_PROD_H = [4, 4, 6, 8, 6, 8, 1, 3]
A2_C1_BOUNDS = np.array([
    [1e-6, 2],
    [1e-6, 1],
    [1e-6, 1],
    [1e-6, 4],  # [1e-6,2],
    [1e-6, 4],  # [1e-6,2],
    [1e-6, 1],
    [6.94E+25/10, 6.94E+25*10], [-2.59, -2.57], [83197*0.9, 83197*1.1],
    [1.53E-01/10, 1.53E-01*10], [4.76/2, 4.76*2], [1294.9*0.9, 1294.9*1.1],
    [3.17E-08/2, 3.17E-08*2], [5.95/2, 5.95*2], [7748*0.9, 7748*1.1],
    [2.59E+09/2, 2.59E+09*2], [1.32/2, 1.32*2], [-0.001, 0.001],
    [1.57E+14/2, 1.57E+14*2], [0.00, 0.06*1.1], [47557*0.9, 47557*1.1],
    [1.59E+04/2, 1.59E+04*2], [2.77, 2.94*1.1], [14882.7*0.9, 14882.7*1.1],
    [3.61E+01/2, 3.61E+01*2], [2.50, 3.86*1.1], [727.4*0.9, 727.4*1.1]])


def x_to_params(x, bounds=A2_C1_BOUNDS):
    # range of bounds. u-l
    r_b = bounds[:, 1]-bounds[:, 0]
    params = x*r_b+bounds[:, 0]
    return params


def params_to_x(params, bounds=A2_C1_BOUNDS):
    ''' normalize parameters to [0, 1] '''
    r_b = bounds[:, 1]-bounds[:, 0]
    x = (params-bounds[:, 0])/r_b
    return x


class CONDITION:
    ''' class for experimental conditions '''

    def __init__(self, T=None, P=None, MF={None: None}):
        ''' inputs:
            T: temperature (scalar)
            P: pressure (scalar)
            MF: mole fractions (dict)
        '''
        self._T = T
        self._P = P
        self._MF = MF
        return

    def T(self):
        return self._T

    def P(self):
        return self._P

    def MF(self):
        return self._MF

    def __repr__(self):
        '''' print all conditions '''
        out = f"Temperature: {self._T} K\nPressure: {self._P} atm\n"
        for k, v in self._MF.items():
            out += f"{k}: {v} (fraction)\n"
        return out


class HYCHEM_A2:
    ''' class holding hychem parameters and generates mechanism file lines '''

    def __init__(self, input_dir, working_dir, chemkin):
        ''' dir: working directory with chem.inp part files '''
        input_dir = fix_dir(input_dir)
        with open(input_dir+"chem_part_1.inp", "r") as f:
            self.chem_part_1 = f.read()
        with open(input_dir+"chem_part_3.inp", "r") as f:
            self.chem_part_3 = f.read()
        self.working_dir = fix_dir(working_dir)
        self.chemkin = chemkin

    def __repr__(self):
        if self.x:
            return self.build_hychem_rxn(self.x)
        else:
            return ""

    def build_hychem_rxn(self, params):
        ''' write HyChem lines in mechanism file '''
        alpha = params[0]
        beta = params[1]
        gamma = params[2]
        lambda_3 = params[3]
        lambda_4 = params[4]
        chi = params[5]
        A1 = params[6]
        m1 = params[7]
        E1 = params[8]
        H_abs_K = params[9:].reshape((-1, 3))

        b_d = (2.+2.*N_C-N_H)/6.
        e_d = (N_C+alpha-2.-(7.-chi)*b_d)/(2.+3.*lambda_3+4*lambda_4)
        b_a = (2.+2.*N_C+2.*gamma-N_H)/6.
        e_a = (N_C-gamma-(7.-chi)*b_a-(1-beta))/(2.+3.*lambda_3+4*lambda_4)

        decomp_coef = [e_d, e_d*lambda_3, e_d*lambda_4, b_d*chi, b_d*(1-chi),
                       alpha, 2-alpha]
        decomp_coef = list(map(lambda x: round(x, 6), decomp_coef))
        oxy_coef = [gamma, e_a, e_a*lambda_3, e_a*lambda_4,
                    b_a*chi, b_a*(1-chi), beta, 1-beta]
        oxy_coef = list(map(lambda x: round(x, 6), oxy_coef))
        # write HyChem lines
        lines = ''
        lines += FUEL_NAME+'=>'
        lines += "".join([str(np.round(decomp_coef[i], 6))+DECOMP_PROD[i]+"+"
                          for i in range(len(DECOMP_PROD))])
        lines = lines[:-1]+" {:e} {:e} {:e}\n".format(A1, m1, E1)
        for i in range(H_abs_K.shape[0]):
            A = H_abs_K[i, 0]
            m = H_abs_K[i, 1]
            E = H_abs_K[i, 2]
            # for A2 / jet fuels
            lines += FUEL_NAME+'+'+SIX_R[i]+'=>'+SIX_RH[i]+'+' + "".join(
                [str(np.round(oxy_coef[i], 6))+H_ABS_PROD[i]+"+"
                 for i in range(len(H_ABS_PROD))])
            lines = lines[:-1]+" {:e} {:e} {:e}\n".format(A, m, E)
        return lines

    def write_cheminp(self, x, bounds=A2_C1_BOUNDS):
        ''' write chem.inp file
            inputs:
            params: (d,1)
            bounds (l,u): (d,2)
        '''
        dir = self.working_dir
        assert(x.shape[0] == bounds.shape[0])
        assert(2 == bounds.shape[1])
        outputs = ''
        outputs += self.chem_part_1
        params = x_to_params(x, bounds)
        # print(params)
        outputs += self.build_hychem_rxn(params)
        # with open(dir+"chem_part_3_no_oxygen.inp", "r") as f:
        outputs += self.chem_part_3
        # print("write_cheminp: "+dir+"chem.inp")
        with open(dir+"chem.inp", "w") as f:
            f.write(outputs)
        return

    def simulate(self, cond=CONDITION()):
        d = self.chemkin.chemkin_wrapper(cond)
        return d

    def _loss(self, x):
        d, cond = self.d, self.cond
        self.write_cheminp(x)
        sim = self.simulate(cond)  # HyChem simulated

        t = d['t']
        tp = sim['t']
        common_s = list(set(d).intersection(set(sim)).difference(set(["t"])))
        fp = sim[common_s]
        interp_f = interp1d(tp, fp, axis=0, fill_value="extrapolate")
        f = interp_f(t)
        # mean squared error
        cost = np.linalg.norm(f-d[common_s])**2./f.size
        # mean absolute error
        # cost = np.sum(np.abs(f-d[common_s]))
        return cost

    def loss(self, x, d, cond):
        ''' loss of HyChem parameters x against data d at condition cond '''
        self.x, self.d, self.cond = x.copy(), d.copy(), cond
        self.cost = self._loss(x)
        return self.cost

    def grad(self, idx, perturb=1e-3):
        ''' gradient of parameters at indices idx '''
        out = np.zeros_like(self.x)
        for i in idx:
            newx = self.x.copy()
            newx[i] = self.x[i]+perturb
            # if newx[i] > 1:
            #     newx[i] = self.x[i]*(1-perturb)
            newcost = self._loss(newx)
            out[i] = (newcost-self.cost)/(newx[i]-self.x[i])
        return out

    def plot(self, names=[], filepath="", log="", title=""):
        self.chemkin.plot_outputs(names=names, filepath=filepath,
                                  log=log, title=title)


class OPTIMIZER:
    ''' class for optimization of hychem parameters '''

    def __init__(self, hychem, chemkin, algo):
        ''' inputs
            hychem: HyChem parameters
            chemkin: CHEKMIN wrapper
        '''
        self.hychem = hychem
        self.chemkin = chemkin
        self.algo = algo
        return

    def optimize(self):
        ''' optimize '''
        if self.algo == "coordinate_desc":
            # coordinate descent
            self._coordinate_desc()
        else:
            raise ValueError("invalid algorithm!")

    def _coordinate_desc(self):
        ''' coordinate descent '''
        raise NotImplementedError("not implemented!")
