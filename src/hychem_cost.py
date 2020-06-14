import numpy as np
from scipy.special import expit
from chemkin_wrapper import chemkin_wrapper
from scipy.interpolate import interp1d
from multiprocessing import Pool
import matplotlib.pyplot as plt
from utils import fix_dir

N_C = 7
N_H = 15
FUEL_NAME = 'SRC7H15'
RAD_NAME = 'SRC7H14'
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
    [3.61E+01/2, 3.61E+01*2], [2.50, 3.86*1.1], [727.4*0.9, 727.4*1.1],
    [4.49E+12/10, 4.49E+12*10], [-0.001, 0.001], [26983.7*0.9, 26983.7*1.1]])


def pcauchy(x):
    return 1/np.pi*np.arctan(x)+.5


def invpcauchy(p):
    return np.tan(np.pi*(p-.5))


def x_to_params(x, bounds):
    # range of bounds. u-l
    r_b = bounds[:, 1]-bounds[:, 0]
    params = expit(x)*r_b+bounds[:, 0]
    # params = pcauchy(x)*r_b+bounds[:,0]
    return params


def params_to_x(params, bounds):
    r_b = bounds[:, 1]-bounds[:, 0]
    x = (params-bounds[:, 0])/r_b
    return np.log(x)-np.log(1-x)
    # return invpcauchy(x)


def build_hychem_rxn(params):
    # assert(params.shape[0]==27)
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    lambda_3 = params[3]
    lambda_4 = params[4]
    chi = params[5]
    A1 = params[6]
    m1 = params[7]
    E1 = params[8]
    A8 = params[27]
    m8 = params[28]
    E8 = params[29]
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
    # print(sum([H_ABS_PROD_C[i]*oxy_coef[i] for i in range(len(oxy_coef))]))
    # print(1+sum([H_ABS_PROD_H[i]*oxy_coef[i] for i in range(len(oxy_coef))]))
    lines = ''
    lines += FUEL_NAME+'=>'
    lines += "".join([str(np.round(decomp_coef[i], 6))+DECOMP_PROD[i]+"+"
                      for i in range(len(DECOMP_PROD))])
    lines = lines[:-1]+" {:e} {:e} {:e}\n".format(A1, m1, E1)
    for i in range(H_abs_K.shape[0]-1):
        A = H_abs_K[i, 0]
        m = H_abs_K[i, 1]
        E = H_abs_K[i, 2]
        # for Shell M
        lines += FUEL_NAME+'+'+SIX_R[i]+'=>'+SIX_RH[i]+'+'+RAD_NAME
        # for A2 / jet fuels
        # lines += FUEL_NAME+'+'+SIX_R[i]+'=>'+SIX_RH[i]+'+'\
        #     +"".join([str(np.round(oxy_coef[i],6))+H_ABS_PROD[i]+"+" \
        #     for i in range(len(H_ABS_PROD))])
        lines = lines+" {:e} {:e} {:e}\n".format(A, m, E)
    lines += RAD_NAME+'=>' \
        + "".join([str(np.round(oxy_coef[i], 6))+H_ABS_PROD[i]+"+"
                   for i in range(len(H_ABS_PROD))])
    lines = lines[:-1]+" {:e} {:e} {:e}\n".format(A8, m8, E8)
    return lines

# take in
# params: (d,1)
# bounds (l,u): (d,2)


def write_cheminp(dir, x, bounds=A2_C1_BOUNDS):
    dir = fix_dir(dir)
    assert(x.shape[0] == bounds.shape[0])
    assert(2 == bounds.shape[1])
    outputs = ''
    with open(dir+"chem_part_1.inp", "r") as f:
        outputs += f.read()
    params = x_to_params(x, bounds)
    # print(params)
    outputs += build_hychem_rxn(params)
    # with open(dir+"chem_part_3_no_oxygen.inp", "r") as f:
    with open(dir+"chem_part_3.inp", "r") as f:
        outputs += f.read()
    # print("write_cheminp: "+dir+"chem.inp")
    with open(dir+"chem.inp", "w") as f:
        f.write(outputs)
    return


def hychem_cost(dir, x, d, cond):
    dir = fix_dir(dir)
    # print("hychem_cost", dir)
    write_cheminp(dir, x)
    sim = chemkin_wrapper(working_dir=dir, **cond)  # HyChem simulated

    t = d['t']
    tp = sim['t']
    common_s = list(set(d).intersection(set(sim)).difference(set(["t"])))
    fp = sim[common_s]
    interp_f = interp1d(tp, fp, axis=0, fill_value="extrapolate")
    f = interp_f(t)
    # meas squared error
    cost = np.linalg.norm(f-d[common_s], ord='fro')**2./(f.shape[0]*f.shape[1])
    # cost = np.sum(np.abs(f-d[common_s]))
    return cost


def grad_hychem_gmm(
        dir, params, cond, perturb=1e-3,
        target_species=['C2H4', 'C3H6', 'C4H81', 'CH4', 'POSF10325', 'H2'],
        target_times=np.arange(1, 11)/10*2e3):
    dir = fix_dir(dir)
    x = params_to_x(params, A2_C1_BOUNDS)
    N_grad = 15

    write_cheminp(dir, x)
    sim = chemkin_wrapper(dir, **cond)
    tp = sim['t']
    fp = sim[target_species]
    interp_f = interp1d(tp, fp, axis=0, fill_value="extrapolate")
    f = interp_f(target_times)
    grad = np.zeros((N_grad, target_times.shape[0]*len(target_species)))
    for i in range(N_grad):
        x1 = np.array(x, copy=True)
        x1[i] = x[i]+perturb
        write_cheminp(dir, x1)
        sim1 = chemkin_wrapper(dir, **cond)

        tp = sim1['t']
        fp = sim1[target_species]
        interp_f = interp1d(tp, fp, axis=0, fill_value="extrapolate")
        f1 = interp_f(target_times)
#         print(f1.shape, f.shape)
        grad_i = (f1-f)/perturb

        # grad_i = (sim1[target_species]-sim[target_species])/perturb
        grad_i = np.reshape(grad_i, (1, -1), order="F")
#         print(grad_i.shape)
        grad[i, :] = grad_i
    return grad


def idt_cost(dir, x, idt, cond):
    dir = fix_dir(dir)
    write_cheminp(dir, x)
    sim = chemkin_wrapper(dir, **cond, TIME=3*idt,
                          DELT=max(1E-6, idt/1e3/100), mode="UV")
    t = sim['t']
    OH = sim['OH']
    idt_sim = t[OH.idxmax()]
    # cost = np.abs((idt-idt_sim)/idt)
    cost = np.abs(np.log(idt_sim/idt))
    return cost, idt_sim


def grad_x_i(dir, x, d, i, perturb, cond, cost):
    # print("grad_x_i", dir)
    x1 = np.array(x, copy=True)
    x1[i] = x[i]+perturb
    cost1 = hychem_cost(dir, x1, d, cond)
    grad = (cost1-cost)/perturb
    # !!!!!!!!! should divide by perturb here!!!!!!!!
    return grad


def grad_idt_i(dir, x, idt, i, perturb, cond, cost):
    x1 = np.array(x, copy=True)
    x1[i] = x[i]+perturb
    cost1, _ = idt_cost(dir, x1, idt, cond)
    grad = (cost1-cost)/perturb
    # !!!!!!!!! should divide by perturb here!!!!!!!!
    return grad


class par_helper:
    def __init__(self, dir, x, d, perturb, cond, cost):
        self.dir = dir
        self.x = x
        self.d = d
        self.perturb = perturb
        self.cost = cost
        self.cond = cond

    def __call__(self, i):
        return grad_x_i(self.dir+str(i)+"/",
                        self.x, self.d, i, self.perturb, self.cond, self.cost)


class par_idt_helper:
    def __init__(self, dir, x, idt, perturb, cond, cost):
        self.dir = dir
        self.x = x
        self.idt = idt
        self.perturb = perturb
        self.cost = cost
        self.cond = cond

    def __call__(self, i):
        return grad_idt_i(self.dir+str(i)+"/",
                          self.x, self.idt, i,
                          self.perturb, self.cond, self.cost)


class grad_hychem_cost_pool:
    def __init__(self, N_thread):
        self.p = Pool(N_thread)

    def __call__(self, dir, x, d, cond, idx=range(15), perturb=.001):
        cost = hychem_cost(dir, x, d, cond)
        grad = self.p.map(par_helper(dir, x, d, perturb, cond, cost), idx)
        # grad = np.zeros(len(idx))
        # for i in idx:
        #     print(i, "out of", len(idx))
        #     func = par_helper(dir,x,d,perturb,cond,cost)
        #     grad[i] = func(i)
        return cost, np.array(grad)

    def __del__(self):
        self.p.close()


class grad_idt_cost_pool:
    def __init__(self, N_thread):
        self.p = Pool(N_thread)

    def __call__(self, dir, x, idt, cond, idx=range(15, 27), perturb=.001):
        cost, _ = idt_cost(dir, x, idt, cond)
        grad = self.p.map(
            par_idt_helper(dir, x, idt, perturb, cond, cost), idx)
        N = x.shape[0]
        grad += [0]*(N-len(idx))
        return cost, np.array(grad)

    def __del__(self):
        self.p.close()


def grad_hychem_cost(dir, x, d, cond, perturb=.001, idx=range(15)):
    if dir[-1] != '/':
        dir += '/'

    p = Pool(4)
    cost = hychem_cost(dir, x, d, cond)
    N = x.shape[0]
    grad = p.map(par_helper(dir, x, d, perturb, cond, cost), idx)
    grad += [0]*(N-len(idx))
    # grad = list(map(par_helper(dir,x,d,perturb), range(x.shape[0])))
    p.close()
    return cost, np.array(grad)


def plot_fitted(x, training_data, metadata, name=""):
    np.random.seed(12312312)
    N = len(training_data)
    Tlist = metadata["Tlist"].tolist()
    Plist = metadata["Plist"].tolist()
    initmf = metadata["initmf"].tolist()
    for i in range(N):
        d = training_data[i]
        write_cheminp("../working_dir/", x)
        cond = {'T': Tlist[i], 'P': Plist[i], 'MF': initmf[i]}
        sim = chemkin_wrapper("../working_dir/", **cond)  # HyChem simulated
        if name != "":
            sim[["t", "SRC7H15", "CH4", "C2H4", "C3H6", "iC4H8"]]\
                .to_csv("../debug_figures/"+name+"_{:d}K_{:.3f}atm.csv"
                        .format(Tlist[i], Plist[i]))
        t = d['t']
        tp = sim['t']
        common_s = list(set(d).intersection(set(sim)).difference(set(["t"])))
        fp = sim[common_s]
        # plot the fitted model
        plt.figure(i)
        plt.gca().set_prop_cycle(None)
        plt.plot(t, d[common_s], '.')
        plt.gca().set_prop_cycle(None)
        plt.plot(tp, fp, '--')
        plt.xlim([0, max(t)])
        # plt.ylim([0, 1.5*np.amax(np.amax(d.iloc[:, 1:]))]);
        plt.ylim([0, 1.05*np.amax(d.iloc[:, 1:].to_numpy())])
        plt.title('T='+str(Tlist[i])+'K, P='+str(Plist[i])+'atm')
        plt.xlabel('Time [us]')
        plt.ylabel('MF')
        plt.legend(common_s)
        if name != "":
            plt.savefig("../debug_figures/"+name+"_{:d}K_{:.3f}atm.pdf"
                        .format(Tlist[i], Plist[i]))


if __name__ == "__main__":
    x = np.zeros(A2_C1_BOUNDS.shape[0])
    hychem_cost("./", x, d)
