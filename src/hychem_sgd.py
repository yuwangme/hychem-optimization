import hychem_cost as hc
import numpy as np
import matplotlib.pyplot as plt
import time


def init_x(init_type):
    if init_type == "shell":
        # initialization of x
        x = hc.params_to_x(np.array([0.1, 0.7, 0.1, 0.5, 0.5, 0.5,
                                     6.94E+25, -2.58, 83197.0,
                                     1.53E-01, 4.76, 1294.9,
                                     3.17E-08, 5.95, 7748.4,
                                     2.59E+09, 1.32, 0.0,
                                     1.57E+14, 0.06, 47557.1,
                                     1.59E+04, 2.93, 14882.7,
                                     3.61E+01, 3.92, 727.4,
                                     4.49E+12, 0.00, 26983.7]), hc.A2_C1_BOUNDS)
    elif init_type == "prev":
        # initialization of x (best x from a previous train)
        x = np.array([1.19E+00, 2.87E+00, -3.69E+00, -1.98E+00, -1.40E+00, 7.90E-01,
                      1.29E+00, -4.44E-14, -1.78E-15, 1.79E+00, -6.93E-01, -6.93E-01,
                      -6.81E-01, -6.93E-01, -6.93E-01, -7.01E-01, -6.93E-01, 0.00E+00,
                      -6.92E-01, 2.30E+00, 1.16E+00, -6.92E-01, -6.42E-01, -6.93E-01,
                      -7.05E-01, 1.47E+00, -2.29E+00, -7.91E-01, 0.00E+00, -1.33E-15])
    else:
        raise ValueError("Invalid init_type!")

    return x


def hychem_sgd(
    training_data,   # training dataset
    metadata,        # metadata: T, P, mole fractions
    idx_range,       # range of indexes to change
    epoch_max,  # number of epochs
    x,          # initial x
    learning_rate,   # learning rate default learning_rate = .75e3
):
    # stochastic gradient descent
    np.random.seed(1231231)
    ghcp = hc.grad_hychem_cost_pool(7)  # use 7 cores

    N_td = len(training_data)  # number of training data points
    Tlist = metadata["Tlist"].tolist()
    Plist = metadata["Plist"].tolist()
    initmf = metadata["initmf"].tolist()

    x_per_iter = np.zeros((epoch_max*N_td, x.shape[0]))  # x in every iteration
    cost = 0  # cost for each epoch
    costs = np.zeros(epoch_max)
    cache = np.zeros_like(x)  # cache for adagrad
    decay_rate = .999
    eps = 1e-3

    start_time = time.time()
    np.random.seed(13123)
    for epoch in range(epoch_max):
        cost = 0
        order = np.random.permutation(N_td)
        for iter in range(len(order)):
            i = order[iter]
            x_per_iter[iter+epoch*N_td, :] = x
            d = training_data[i]
            cond = {'T': Tlist[i], 'P': Plist[i], 'MF': initmf[i]}
            cost_i, grad = ghcp("../working_dir/", x, d, cond, idx=idx_range)
            # adagrad (adaptive gradient)
            cache[idx_range] += grad**2
            delta = grad*learning_rate/(np.sqrt(cache[idx_range])+eps)
            x[idx_range] -= delta
            # RMSprop
            # cache = decay_rate * cache + (1 - decay_rate) * grad**2
            # x -= learning_rate * grad / (np.sqrt(cache) + eps)

            cost = cost+cost_i
            print("\titer", iter, Tlist[i], "K", "\ts(ci)", "{:.3e}".format(np.sqrt(cost_i)),
                  "\tdeltaL1 {:.3e}".format(np.linalg.norm(delta, 1)))
        costs[epoch] = cost/N_td
        if epoch == 0:
            cost0 = costs[epoch]
        print("epoch", epoch, "\tcost", "{:.3f}".format(np.sqrt(costs[epoch]/cost0)))
    end_time = time.time()
    print("Runtime: ", round((end_time-start_time)/60, 2), " min")
    return x_per_iter, costs


def plot_iter(x_per_iter, idx_range, costs):
    names = ["0", "1", "2", "3", "4", "5"]
    for i in range(8):
        names += [j+str(i) for j in ["A", "b", "E"]]
    plt.figure()
    plt.plot(range(x_per_iter.shape[0]), x_per_iter[:, idx_range[idx_range <= 5]], '--')
    plt.plot(range(x_per_iter.shape[0]), x_per_iter[:, idx_range[idx_range > 5]])
    plt.legend(labels=[names[i] for i in idx_range], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.plot(range(x_per_iter.shape[0]), [-3]*x_per_iter.shape[0], 'b-.')
    plt.plot(range(x_per_iter.shape[0]), [+3]*x_per_iter.shape[0], 'b-.')
    plt.xlabel("iteration")
    plt.ylabel("transformed parameters")
    plt.tight_layout()
    plt.savefig("../debug_figures/x_vs_iter.pdf")

    plt.figure()
    plt.plot(range(costs.shape[0]), (costs/costs[0])**.5, '-o')
    plt.yscale("log")
    plt.xlabel("epoch (pass through all data)")
    plt.ylabel("cost")
    plt.tight_layout()
    plt.savefig("../debug_figures/cost_vs_epoch.pdf")

    np.savetxt("../debug_figures/x_per_iter.csv", x_per_iter[:, idx_range], delimiter=",",
               header="".join([names[i]+"," for i in idx_range])[:-1], comments='')
