#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from sklearn import metrics
from tensorflow.keras import models
from ortools.linear_solver import pywraplp
import eml.backend.ortool_backend as ortools_backend
from eml.net.reader import keras_reader
from eml.net.process import ibr_bounds
from eml.net.embed import encode

def plot_df_cols(data, figsize=None, legend=True):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    x = data.index
    plt.xlabel(data.index.name)
    # Plot all columns
    for cname in data.columns:
        y = data[cname]
        plt.plot(x, y, label=cname)
    # Add legend
    if legend and len(data.columns) <= 10:
        plt.legend(loc='best')
    plt.grid(':')
    # Make it compact
    plt.tight_layout()
    # Show
    plt.show()


def SIR(y, beta, gamma):
    # Unpack the state
    S, I, R = y
    N = sum([S, I, R])
    # Compute partial derivatives
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    # Return gradient
    return np.array([dS, dI, dR])


def simulate_SIR(S0, I0, R0, beta, gamma, tmax, steps_per_day=1):
    # Build initial state
    Z = S0 + I0 + R0
    Z = Z if Z > 0 else 1 # Handle division by zero
    y0 = np.array([S0, I0, R0]) / Z
    # Wrapper
    nabla = lambda y, t: SIR(y, beta, gamma)
    # Solve
    t = np.linspace(0, tmax, steps_per_day * tmax)
    Y = odeint(nabla, y0, t)
    # Wrap as dataframe
    data = pd.DataFrame(data=Y, index=t, columns=['S', 'I', 'R'])
    data.index.rename('time', inplace=True)
    # Return the results
    return data


def plot_2D_samplespace(data, figsize=None):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup axes
    plt.xlabel('x_0')
    plt.xlabel('x_1')
    # Plot points
    plt.plot(data[:, 0], data[:, 1], 'bo', label='samples', color='tab:blue')
    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=60, alpha=0.3,
            color='tab:blue')
    plt.grid(':')
    # Make it compact
    plt.tight_layout()
    # Show
    plt.show()


def sample_points(ranges, n_samples, mode, seed=None):
    assert(mode in ('uniform', 'lhs', 'max_min'))
    # Build a space
    space = Space(ranges)
    # Seed the RNG
    np.random.seed(seed)
    # Sample
    if mode == 'uniform':
        X = space.rvs(n_samples)
    elif mode == 'lhs':
        lhs = Lhs(lhs_type="classic", criterion=None)
        X = lhs.generate(space.dimensions, n_samples)
    elif mode == 'max_min':
        lhs = Lhs(criterion="maximin", iterations=100)
        X = lhs.generate(space.dimensions, n_samples)
    # Convert to an array
    return np.array(X)


def generate_SIR_input(max_samples, mode='lhs',
        normalize=True, seed=None, max_beta=1):
    # Sampling space: unnormalized S, I, R, plus beta
    ranges = [(0.,1.), (0.,1.), (0.,1.), (0., float(max_beta))]
    # Generate input
    X = sample_points(ranges, max_samples, mode, seed=seed)
    # Normalize
    if normalize:
        # Compute the normalization constants
        Z = np.sum(X[:, :3], axis=1)
        Z = np.where(Z > 0, Z, 1)
        # Normalize the first three columns
        X[:, :3] = X[:, :3] / Z.reshape(-1, 1)
    # Wrap into a DataFrame
    data = pd.DataFrame(data=X, columns=['S', 'I', 'R', 'beta'])
    return data


def generate_SIR_output(sir_in, gamma, tmax, steps_per_day=1):
    # Prepare a data structure for the results
    res = []
    # Loop over all examples
    for idx, in_series in sir_in.iterrows():
        # Unpack input
        S0, I0, R0, beta = in_series
        # Simulate
        sim_data = simulate_SIR(S0, I0, R0, beta, gamma, tmax,
                steps_per_day=steps_per_day)
        # Compute output
        res.append(sim_data.values[-1, :])
    # Wrap into a dataframe
    data = pd.DataFrame(data=res, columns=['S', 'I', 'R'])
    # Return
    return data


# def build_nn_model(input_shape, output_shape, hidden,
#         output_activation='linear', kernel_regularizers=[], scale=None, name=None):
#     model_in = keras.Input(shape=input_shape, dtype='float32')
#     x = model_in
#     for i, h in enumerate(hidden):
#         kr = kernel_reguralizers[i] if i < len(kernel_regularizers)-1 else None
#         x = layers.Dense(h, activation='relu', kernel_regularizer=kr)(x)
#     kr_out = kernel_regularizers[-1] if len(kernel_regularizers) > len(hidden) else None
#     model_out = layers.Dense(output_shape, activation=output_activation,
#             kernel_regularizer=kr_out)(x)
#     if scale is not None:
#         model_out *= scale
#     model = keras.Model(model_in, model_out, name=name)
#     return model


def build_nn_model(input_shape, output_shape, hidden=[],
        output_activation='linear', name=None):
    # Build all layers
    ll = [keras.Input(input_shape)]
    for h in hidden:
        ll.append(layers.Dense(h, activation='relu'))
    ll.append(layers.Dense(output_shape, activation=output_activation))
    # Build the model
    model = keras.Sequential(ll, name=name)
    return model


def train_nn_model(model, X, y, loss,
        verbose=0, patience=10,
        validation_split=0.0, **fit_params):
    # Compile the model
    model.compile(optimizer='Adam', loss=loss)
    # Build the early stop callback
    cb = []
    if validation_split > 0:
        cb += [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    # Train the model
    history = model.fit(X, y, callbacks=cb,
            validation_split=validation_split,
            verbose=verbose, **fit_params)
    return history


def plot_nn_model(model, show_layer_names=True, show_layer_activations=True, show_shapes=True):
    return keras.utils.plot_model(model, show_shapes=show_shapes,
            show_layer_names=show_layer_names, rankdir='LR',
            show_layer_activations=show_layer_activations)


def plot_training_history(history=None,
        figsize=None, print_final_scores=True):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    if print_final_scores:
        trl = history.history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history.history:
            vll = history.history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)


def print_ml_metrics(model, X, y, label=None):
    # Obtain the predictions
    pred = model.predict(X, verbose=0)
    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(y, pred))
    # Compute the MAE
    mae = metrics.mean_absolute_error(y, pred)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(y, pred)
    lbl = '' if label is None else f' ({label})'
    print(f'R2: {r2:.2f}, MAE: {mae:.2}, RMSE: {rmse:.2f}{lbl}')


def save_ml_model(model, name):
    model.save_weights(f'../data/{name}.h5')
    with open(f'../data/{name}.json', 'w') as f:
        f.write(model.to_json())


def load_ml_model(name):
    with open(f'../data/{name}.json') as f:
        model = models.model_from_json(f.read())
        model.load_weights(f'../data/{name}.h5')
        return model


class NPI(object):
    """Docstring for NPI. """

    def __init__(self, name, effect, cost):
        """TODO: to be defined. """
        self.name = name
        self.effect = effect
        self.cost = cost


def solve_sir_planning(
        keras_model,
        npis : list,
        S0 : float,
        I0 : float,
        R0 : float,
        beta_base : float,
        nweeks : int = 1,
        budget : float = None,
        tlim : float = None,
        init_state_csts : bool = True,
        network_csts : bool = True,
        effectiveness_csts : bool = True,
        cost_csts : bool = True,
        beta_ub_csts : bool = True,
        use_hints : bool = True):

    assert(0 <= beta_base <= 1)

    # Build a model object
    slv = pywraplp.Solver.CreateSolver('CBC')

    # Define the variables
    X = {}
    for t in range(nweeks+1):
        # Build the SIR variables
        X['S', t] = slv.NumVar(0, 1, f'S_{t}')
        X['I', t] = slv.NumVar(0, 1, f'I_{t}')
        X['R', t] = slv.NumVar(0, 1, f'R_{t}')
        if t < nweeks:
            X['b', t] = slv.NumVar(0, 1, f'b_{t}')

        # Build the NPI variables
        if t < nweeks:
            for npi in npis:
                name = npi.name
                X[name, t] = slv.IntVar(0, 1, f'{name}_{t}')

    # Build the cost variable
    maxcost = sum(npi.cost for npi in npis) * nweeks
    X['cost'] = slv.NumVar(0, maxcost, 'cost')

    # Build the initial state constraints
    if init_state_csts:
        slv.Add(X['S', 0] == S0)
        slv.Add(X['I', 0] == I0)
        slv.Add(X['R', 0] == R0)

    # Build the network constraints
    if network_csts:
        # Build a backend object
        bkd = ortools_backend.OrtoolsBackend()
        # Convert the keras model in internal format
        nn = keras_reader.read_keras_sequential(keras_model)
        # Set bounds
        nn.layer(0).update_lb(np.zeros(4))
        nn.layer(0).update_ub(np.ones(4))
        # Propagate bounds
        ibr_bounds(nn)
        # Build the encodings
        for t in range(1, nweeks+1):
            vin = [X['S',t-1], X['I',t-1], X['R',t-1], X['b',t-1]]
            vout = [X['S',t], X['I',t], X['R',t]]
            encode(bkd, nn, slv, vin, vout, f'nn_{t}')

    # Build the effectiveness constraints
    if effectiveness_csts:
        for t in range(nweeks):
            # Set base beta as the starting one
            bbeta = beta_base
            # Process all NPIs
            for i, npi in enumerate(npis):
                name = npi.name
                # For all NPIs but the last, build a temporary beta variable
                if i < len(npis)-1:
                    # Build a new variable to be used as current beta
                    cbeta = slv.NumVar(0, 1, f'b_{name}_{t}')
                    X[f'b_{name}', t] = cbeta
                else:
                    # Use the "real" beta as current
                    cbeta = X['b', t]
                # Linearize a guarded division
                slv.Add(cbeta >= npi.effect * bbeta - 1 + X[name, t])
                slv.Add(cbeta >= bbeta - X[name, t])
                # Add an upper bound, if requested
                if beta_ub_csts:
                    slv.Add(cbeta <= npi.effect * bbeta + 1 - X[name, t])
                    slv.Add(cbeta <= bbeta + X[name, t])
                # Reset base beta
                bbeta = cbeta

    # Define the cost
    if cost_csts:
        slv.Add(X['cost'] == sum(npi.cost * X[npi.name, t]
            for t in range(nweeks) for npi in npis))
        if budget is not None:
            slv.Add(X['cost'] <= budget)

    # Define the objectives
    slv.Maximize(X['S', nweeks])

    # Build a heuristic solution
    if use_hints:
        hvars, hvals = [], []
        # Sort NPIs by decreasing "efficiency"
        snpis = sorted(npis, key=lambda npi: -(1-npi.effect) / npi.cost)
        # Loop over all the NPIs
        rbudget = budget
        for npi in snpis:
            # Activate on as many weeks as possible
            for w in range(nweeks):
                if rbudget > npi.cost:
                    hvars.append(X[npi.name, w])
                    hvals.append(1)
                    rbudget -= npi.cost
                else:
                    hvars.append(X[npi.name, w])
                    hvals.append(0)
        # Set hints
        slv.SetHint(hvars, hvals)

    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(tlim * 1000)
    # Solve the problem
    status = slv.Solve()
    # Return the result
    res = None
    closed = False
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        res = {}
        for k, x in X.items():
            res[k] = x.solution_value()
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.INFEASIBLE):
        closed = True
    return res, closed


def sol_to_dataframe(sol, npis, nweeks):
    # Define the column names
    cols = ['S', 'I', 'R', 'b']
    cols += [npi.name for npi in npis]
    # Prepare a result dataframe
    res = pd.DataFrame(index=range(nweeks+1), columns=cols, data=np.nan)
    for w in range(nweeks):
        res.loc[w, 'S'] = sol['S', w]
        res.loc[w, 'I'] = sol['I', w]
        res.loc[w, 'R'] = sol['R', w]
        res.loc[w, 'b'] = sol['b', w]
        for n in npis:
            res.loc[w, n.name] = sol[n.name, w]
    # Store the state for the final week
    res.loc[nweeks, 'S'] = sol['S', nweeks]
    res.loc[nweeks, 'I'] = sol['I', nweeks]
    res.loc[nweeks, 'R'] = sol['R', nweeks]
    # Return the result
    return res


def simulate_SIR_NPI(S0, I0, R0, beta, gamma, steps_per_day=1):
    # Prepare the result data structure
    S, I, R = [], [], []
    # Store the initial state
    S.append(S0)
    I.append(I0)
    R.append(R0)
    # Loop over all weeks
    for i, b in enumerate(beta):
        # Simulate one week
        wres = simulate_SIR(S[i], I[i], R[i], b, gamma, 7,
                steps_per_day=steps_per_day)
        # Store the final state
        last = wres.iloc[-1]
        S.append(last['S'])
        I.append(last['I'])
        R.append(last['R'])
    # Wrap into a dataframe
    res = pd.DataFrame(data=np.array([S, I, R]).T,
            columns=['S', 'I', 'R'])
    return res


def solve_sir_brute_force(
        npis : list,
        S0 : float,
        I0 : float,
        R0 : float,
        beta_base : float,
        gamma : float,
        nweeks : int = 1,
        budget : float = None):
    # Build a table with beta-confs for a week
    doms = [np.array([0, 1]) for _ in range(len(npis))]
    beta_confs = cartesian_product(doms)
    # Compute costs
    npi_costs = np.array([npi.cost for npi in npis])
    beta_costs = np.dot(beta_confs, npi_costs)
    # Compute beta values
    npi_eff = np.array([npi.effect for npi in npis])
    beta_vals = beta_confs * npi_eff
    beta_vals += (1-beta_confs)
    beta_vals = beta_base * np.product(beta_vals, axis=1)
    # Build all combinations of beta-values and costs
    bsched_cost = cartesian_product([beta_costs for _ in range(nweeks)])
    bsched_vals = cartesian_product([beta_vals for _ in range(nweeks)])
    # Filter out configurations that do not meen the budget
    mask = np.sum(bsched_cost, axis=1) <= budget
    bsched_feas = bsched_vals[mask]
    # Simulate them all
    best_S = -np.inf
    best_sched = None
    for bsched in bsched_feas:
        # Simulate
        res = simulate_SIR_NPI(S0, I0, R0, bsched, gamma, steps_per_day=1)
        last_S = res.iloc[-1]['S']
        if last_S > best_S:
            best_S = last_S
            best_sched = bsched
    return best_S, best_sched


def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def generate_market_dataset(nsamples, nitems, noise=0, seed=None):
    assert(0 <= noise <= 1)
    # Seed the RNG
    np.random.seed(seed)
    # Generate costs
    speed = np.random.choice([10, 13, 15], size=nitems)
    base = 0.4 * np.random.rand(nitems)
    scale = 0.4 + 1 * np.random.rand(nitems)
    offset = -0.7 * np.random.rand(nitems)

    # Generate input
    x = np.random.rand(nsamples)
    # Prepare a result dataset
    res = pd.DataFrame(data=x, columns=['x'])

    # scale = np.sort(scale)[::-1]
    for i in range(nitems):
        # Compute base cost
        cost = scale[i] / (1 + np.exp(-speed[i] * (x + offset[i])))
        # Rebase
        cost = cost - np.min(cost) + base[i]
        # sx = direction[i]*speed[i]*(x+offset[i])
        # cost = base[i] + scale[i] / (1 + np.exp(sx))
        res[f'C{i}'] = cost
    # Add noise
    if noise > 0:
        for i in range(nitems):
            pnoise = noise * np.random.randn(nsamples)
            res[f'C{i}'] = np.maximum(0, res[f'C{i}'] + pnoise)
    # Reindex
    res.set_index('x', inplace=True)
    # Sort by index
    res.sort_index(inplace=True)
    # Return results
    return res


def train_test_split(data, test_size, seed=None):
    assert(0 < test_size < 1)
    # Seed the RNG
    np.random.seed(seed)
    # Shuffle the indices
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    # Partition the indices
    sep = int(len(data) * (1-test_size))
    res_tr = data.iloc[idx[:sep]]
    res_ts = data.iloc[idx[sep:]]
    return res_tr, res_ts


def get_ml_metrics(model, X, y):
    # Obtain the predictions
    pred = model.predict(X, verbose=0)
    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(y, pred))
    # Compute the MAE
    mae = metrics.mean_absolute_error(y, pred)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(y, pred)
    return r2, mae, rmse


def generate_market_problem(nitems, rel_req, seed=None):
    # Seed the RNG
    np.random.seed(seed)
    # Generate the item values
    # values = np.ones(nitems)
    values = 1 + 0.2*np.random.rand(nitems)
    # Generate the requirement
    req = rel_req * np.sum(values)
    # Return the results
    return MarketProblem(values, req)


class MarketProblem(object):
    """Docstring for MarketProblem. """

    def __init__(self, values, requirement):
        """TODO: to be defined. """
        # Store the problem configuration
        self.values = values
        self.requirement = requirement

    def solve(self, costs, tlim=None, print_solution=False):
        # Quick access to some useful fields
        values = self.values
        req = self.requirement
        nv = len(values)
        # Build the solver
        slv = pywraplp.Solver.CreateSolver('CBC')
        # Build the variables
        x = [slv.IntVar(0, 1, f'x_{i}') for i in range(nv)]
        # Build the requirement constraint
        rcst = slv.Add(sum(values[i] * x[i] for i in range(nv)) >= req)
        # Build the objective
        slv.Minimize(sum(costs[i] * x[i] for i in range(nv)))

        # Set a time limit, if requested
        if tlim is not None:
            slv.SetTimeLimit(1000 * tlim)
        # Solve the problem
        status = slv.Solve()
        # Prepare the results
        if status in (slv.OPTIMAL, slv.FEASIBLE):
            res = []
            # Extract the solution
            sol = [x[i].solution_value() for i in range(nv)]
            res.append(sol)
            # Determine whether the problem was closed
            if status == slv.OPTIMAL:
                res.append(True)
            else:
                res.append(False)
        else:
            # TODO I am not handling the unbounded case
            # It should never arise in the first place
            if status == slv.INFEASIBLE:
                res = [None, True]
            else:
                res = [None, False]
        # Print the solution, if requested
        if print_solution:
            print_sol(self, res[0], res[1], costs)
        return res

    def __repr__(self):
        return f'<MarketProblem: {self.values} {self.requirement}>'


def print_sol(prb, sol, closed, costs):
    # Obtain indexes of selected items
    idx = [i for i in range(len(sol)) if sol[i] > 0]
    # Obtain the corresponding values
    values = [prb.values[i] for i in idx]
    # Print selected items with values and costs
    s = ', '.join(f'{i}' for i in idx)
    print('Selected items:', s)
    s = f'Cost: {sum(costs):.2f}, '
    s += f'Value: {sum(values):.2f}, '
    s += f'Requirement: {prb.requirement:.2f}, '
    s += f'Closed: {closed}'
    print(s)


def compute_regret(problem, predictor, pred_in, true_costs, tlim=None):
    # Obtain all predictions
    costs = predictor.predict(pred_in, verbose=0)
    # Compute all solutions
    sols = []
    for c in costs:
        sol, _ = problem.solve(c, tlim=tlim)
        sols.append(sol)
    sols = np.array(sols)
    # Compute the true solutions
    tsols = []
    for c in true_costs:
        sol, _ = problem.solve(c, tlim=tlim)
        tsols.append(sol)
    tsols = np.array(tsols)
    # Compute true costs
    costs_with_predictions = np.sum(true_costs * sols, axis=1)
    costs_with_true_solutions = np.sum(true_costs * tsols, axis=1)
    # Compute regret
    regret = costs_with_predictions - costs_with_true_solutions
    # Return true costs
    return regret


def plot_histogram(data, label=None, bins=20, figsize=None,
        data2=None, label2=None, print_mean=False):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    plt.xlabel(label)
    # Define bins
    rmin, rmax = data.min(), data.max()
    if data2 is not None:
        rmin = min(rmin, data2.min())
        rmax = max(rmax, data2.max())
    bins = np.linspace(rmin, rmax, bins)
    # Histogram
    hist, edges = np.histogram(data, bins=bins)
    hist = hist / np.sum(hist)
    plt.step(edges[:-1], hist, where='post', label=label)
    if data2 is not None:
        hist2, edges2 = np.histogram(data2, bins=bins)
        hist2 = hist2 / np.sum(hist2)
        plt.step(edges2[:-1], hist2, where='post', label=label2)
    plt.grid(':')
    # Make it compact
    plt.tight_layout()
    # Legend
    plt.legend()
    # Show
    plt.show()
    # Print mean, if requested
    if print_mean:
        s = f'Mean: {np.mean(data):.3f}'
        if label is not None:
            s += f' ({label})'
        if data2 is not None:
            s += f', {np.mean(data2):.3f}'
            if label2 is not None:
                s += f' ({label2})'
        print(s)


def build_dfl_ml_model(input_size, output_size,
        problem, tlim=None, hidden=[], recompute_chance=1,
        output_activation='linear', name=None):
    # Build all layers
    nnin = keras.Input(input_size)
    nnout = nnin
    for h in hidden:
        nnout = layers.Dense(h, activation='relu')(nnout)
    nnout = layers.Dense(output_size, activation=output_activation)(nnout)
    # Build the model
    model = DFLModel(problem, tlim=tlim, recompute_chance=recompute_chance,
            inputs=nnin, outputs=nnout, name=name)
    return model


class DFLModel(keras.Model):
    def __init__(self, prb, tlim=None, recompute_chance=1, **params):
        super(DFLModel, self).__init__(**params)
        # Store configuration parameters
        self.prb = prb
        self.tlim = tlim
        self.recompute_chance = recompute_chance
        # Build metrics
        self.metric_loss = keras.metrics.Mean(name="loss")
        # self.metric_regret = keras.metrics.Mean(name="regret")
        # self.metric_mae = keras.metrics.MeanAbsoluteError(name="mae")
        # Prepare a field for the solutions
        self.sol_store = None

    # def dfl_fit(self, X, y, **kwargs):
    #     # Precompute all solutions for the true costs
    #     self.sol_store = []
    #     for c in y:
    #         sol, closed = self.prb.solve(c, tlim=self.tlim)
    #         self.sol_store.append(sol)
    #     self.sol_store = np.array(self.sol_store)
    #     # Call the normal fit method
    #     return self.fit(X, y, **kwargs)


    def fit(self, X, y, **kwargs):
        # Precompute all solutions for the true costs
        self.sol_store = []
        for c in y:
            sol, closed = self.prb.solve(c, tlim=self.tlim)
            self.sol_store.append(sol)
        self.sol_store = np.array(self.sol_store)
        # Call the normal fit method
        return super(DFLModel, self).fit(X, y, **kwargs)

    def _find_best(self, costs):
        tc = np.dot(self.sol_store, costs)
        best_idx = np.argmin(tc)
        best = self.sol_store[best_idx]
        return best

    def train_step(self, data):
        # Unpack the data
        x, costs_true = data
        # Quick access to some useful fields
        prb = self.prb
        tlim = self.tlim

        # Loss computation
        with tf.GradientTape() as tape:
            # Obtain the predictions
            costs = self(x, training=True)
            # Solve all optimization problems
            sols, tsols = [], []
            for c, tc in zip(costs.numpy(), costs_true.numpy()):
                # Decide whether to recompute the solution
                if np.random.rand() < self.recompute_chance:
                    sol, closed = prb.solve(c, tlim=self.tlim)
                    # Store the solution, if needed
                    if self.recompute_chance < 1:
                        # Check if the solutions is already stored
                        if not (self.sol_store == sol).all(axis=1).any():
                            self.sol_store = np.vstack((self.sol_store, sol))
                else:
                    sol = self._find_best(c)
                # Find the best solution with the predicted costs
                sols.append(sol)
                # Find the best solution with the true costs
                tsol = self._find_best(tc)
                tsols.append(tsol)
            # Convert solutions to numpy arrays
            sols = np.array(sols)
            tsols = np.array(tsols)
            # Compute the cost difference
            cdiff = costs - costs_true
            # Compute the solution difference
            sdiff = tsols - sols
            # Compute the loss terms
            loss_terms = tf.reduce_sum(cdiff * sdiff, axis=1)
            # Compute the mean loss
            loss = tf.reduce_mean(loss_terms)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update main metrics
        self.metric_loss.update_state(loss)
        # regrets = tf.reduce_sum((sols - tsols) * costs_true, axis=1)
        # mean_regret = tf.reduce_mean(regrets)
        # self.metric_regret.update_state(mean_regret)
        # self.metric_mae.update_state(costs_true, costs)
        # Update compiled metrics
        self.compiled_metrics.update_state(costs_true, costs)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    # def test_step(self, data):
    #     # Unpack the data
    #     x, costs_true = data
    #     # Compute predictions
    #     costs = self(x, training=False)
    #     # Updates the metrics tracking the loss
    #     self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #     # Update the metrics.
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value.
    #     # Note that it will include the loss (tracked in self.metrics).
    #     return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.metric_loss]


def train_dfl_model(model, X, y, tlim=None,
        epochs=20, verbose=0, patience=10, batch_size=32,
        validation_split=0.2):
    # Compile and train
    model.compile(optimizer='Adam', run_eagerly=True)
    if validation_split > 0:
        cb = [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    else:
        cb = None
    # history = model.dfl_fit(X, y, validation_split=validation_split,
    #                  callbacks=cb, batch_size=batch_size,
    #                  epochs=epochs, verbose=verbose)
    history = model.fit(X, y, validation_split=validation_split,
                     callbacks=cb, batch_size=batch_size,
                     epochs=epochs, verbose=verbose)
    return history
