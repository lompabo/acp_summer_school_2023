import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn import metrics
from ortools.linear_solver import pywraplp
import tensorflow as tf
from scipy import stats
import tensorflow_probability as tfp

# ==============================================================================
# Loss Function Analysis
# ==============================================================================

def eval_ground_truth_(x):
    p0=[0., 0., 2.5]
    p1=[0.3, 0.8, 0]
    f0 = np.polynomial.Polynomial(p0)
    f1 = np.polynomial.Polynomial(p1)
    res = np.stack((f0(x), f1(x)), axis=-1)
    return res


def draw_ground_truth_(x):
    y = eval_ground_truth_(x)
    plt.plot(x, y[:, 0], label=r'$y_0$', color='0.5', linestyle='-')
    plt.plot(x, y[:, 1], label=r'$y_1$', color='0.5', linestyle=':')


def eval_predictions_(x, w, model=1):
    assert(model in (1, 2, 3))
    if model == 1:
        c0 = w**2 * x
        c1 = 0.5 * w * x**0
        # c1 = 0.5 * np.abs(w) * x**0
    elif model == 2:
        c0 = w * x
        c1 = 1 - c0
        # c0 = w * x
        # c1 = 0.5 * x**0
    elif model == 3:
        c0 = w * x
        c1 = 0.1 - c0
    res = np.stack((c0, c1), axis=-1)
    return res


def find_optimal_w_(x, model=1):
    y = eval_ground_truth_(x)
    wvals = np.linspace(0, 2, 1000)
    mse = []
    for w in wvals:
        yp = eval_predictions_(x, w, model=model)
        mse.append(np.mean(np.square(yp - y)))
    return wvals[np.argmin(mse)]


def draw_predictions_(x, w, model=1):
    y = eval_predictions_(x, w, model=model)
    plt.plot(x, y[:, 0], label=r'$\hat{y}_0$', color='tab:orange', linestyle='-')
    plt.plot(x, y[:, 1], label=r'$\hat{y}_1$', color='tab:orange', linestyle=':')


def draw(xmin=0, xmax=1, npoints=100, w=None, model=1, figsize=(10, 5)):
    x = np.linspace(xmin, xmax, npoints)
    if w is None:
        w = find_optimal_w_(x, model=model)
        print(f'Optimized theta: {w:.3f}')
    plt.figure(figsize=figsize)
    draw_ground_truth_(x)
    draw_predictions_(x, w)
    plt.grid(':')
    plt.legend()
    plt.show()


def normal_sample_(mean, std, size=1, seed=42):
    np.random.seed(seed)
    return norm.rvs(size=size, loc=mean, scale=std)


def get_decisions_(c):
    # Compute the optimal decisions
    x_0 = c[:, :, 0] <= c[:, :, 1]
    x = np.stack((x_0, 1 - x_0), axis=-1)
    return x

# def get_estimated_costs_and_decisions_(w, x, use_spo_sol=False):
#     # Compute the estimated costs (just for arc 0)
#     c_hat = eval_predictions_(x, w)
#     # Compute the decisions
#     if use_spo_sol:
#         x_hat_0 = (2*c_hat[:, 0] - c_star[:, 0]) < (2*c_hat[:, 1] - c_star[:, 1])
#     else:
#         x_hat_0 = c_star[:, 0] < c_star[:, 1]
#     y_hat = np.vstack((x_hat_0, 1 - x_hat_0)).T
#     return c_hat, y_hat


# def loss_regret_(c_star, c_hat, y_star, y_hat):
#     costs = np.sum(c_star * y_hat, axis=-1)
#     best_costs = np.sum(c_star * y_star, axis=-1)
#     return np.mean(costs - best_costs)


# def loss_self_contrastive_(c_star, c_hat, y_star, y_hat):
#     best_est_costs = np.sum(c_hat * y_hat, axis=-1)
#     est_best_costs = np.sum(c_hat * y_star, axis=-1)
#     return np.mean(est_best_costs - best_est_costs)


# def loss_spo_(c_star, c_hat, y_star, y_spo, alpha):
#     c_spo = alpha * c_hat - c_star
#     best_spo_costs = np.sum(c_spo * y_spo, axis=-1)
#     spo_best_costs = np.sum(c_spo * y_star, axis=-1)
#     return np.mean(spo_best_costs - best_spo_costs)


# def loss_self_constrastive_(w, x):
#     # Obtain costs and decision
#     c_star, c_hat, y_star, y_hat = get_costs_and_decisions_(w, x)
#     # Compute the actual cost of the decisions
#     pred_costs = np.choose(y_hat, c_hat.T)
#     contr_costs = np.choose(y_star, c_hat.T)
#     return np.mean(contr_costs - pred_costs)

class ActualCostLoss:
    def __init__(self, smoothing_samples=0, smoothing_std=0.05, seed=42):
        self.smoothing_samples = smoothing_samples
        self.smoothing_std = smoothing_std
        if smoothing_samples > 0:
            self.noise = normal_sample_(mean=0, std=smoothing_std, size=(smoothing_samples, 1, 2), seed=seed)
        else:
            self.noise = np.zeros((1, 1, 2))

    def __call__(self, c_star, c_hat, y_star):
        c_hat = c_hat + self.noise
        y_hat = get_decisions_(c_hat)
        costs = np.sum(c_star * y_hat, axis=-1)
        return np.mean(costs)

    def __repr__(self):
        res = 'actual cost'
        if self.smoothing_samples > 0:
            res += f' (ns={self.smoothing_samples}, std={self.smoothing_std})'
        return res


class RegretLoss:
    def __init__(self, smoothing_samples=0, smoothing_std=0.05, seed=42):
        self.smoothing_samples = smoothing_samples
        self.smoothing_std = smoothing_std
        if smoothing_samples > 0:
            self.noise = normal_sample_(mean=0, std=smoothing_std, size=(smoothing_samples, 1, 2), seed=seed)
        else:
            self.noise = np.zeros((1, 1, 2))

    def __call__(self, c_star, c_hat, y_star):
        c_hat = c_hat + self.noise
        y_hat = get_decisions_(c_hat)
        costs = np.sum(c_star * y_hat, axis=-1)
        best_costs = np.sum(c_star * y_star, axis=-1)
        return np.mean(costs - best_costs)

    def __repr__(self):
        res = 'regret'
        if self.smoothing_samples > 0:
            res += f' (ns={self.smoothing_samples}, std={self.smoothing_std})'
        return res


class SelfContrastiveLoss:
    def __init__(self, smoothing_samples=0, smoothing_std=0.05, seed=42):
        self.smoothing_samples = smoothing_samples
        self.smoothing_std = smoothing_std
        if smoothing_samples > 0:
            self.noise = normal_sample_(mean=0, std=smoothing_std, size=(smoothing_samples, 1, 2), seed=seed)
        else:
            self.noise = np.zeros((1, 1, 2))

    def __call__(self, c_star, c_hat, y_star):
        c_hat = c_hat + self.noise
        y_hat = get_decisions_(c_hat)
        best_est_costs = np.sum(c_hat * y_hat, axis=-1)
        est_best_costs = np.sum(c_hat * y_star, axis=-1)
        return np.mean(est_best_costs - best_est_costs)

    def __repr__(self):
        res = 'self contrastive'
        if self.smoothing_samples > 0:
            res += f' (ns={self.smoothing_samples}, std={self.smoothing_std})'
        return res


class SelfContrastiveRegretLoss:
    def __init__(self, smoothing_samples=0, smoothing_std=0.05, seed=42):
        self.smoothing_samples = smoothing_samples
        self.smoothing_std = smoothing_std
        if smoothing_samples > 0:
            self.noise = normal_sample_(mean=0, std=smoothing_std, size=(smoothing_samples, 1, 2), seed=seed)
        else:
            self.noise = np.zeros((1, 1, 2))

    def __call__(self, c_star, c_hat, y_star):
        c_hat = c_hat + self.noise
        y_hat = get_decisions_(c_hat)
        costs = np.sum(c_star * y_hat, axis=-1)
        best_costs = np.sum(c_star * y_star, axis=-1)
        best_est_costs = np.sum(c_hat * y_hat, axis=-1)
        est_best_costs = np.sum(c_hat * y_star, axis=-1)
        return np.mean((est_best_costs - best_est_costs) + (costs - best_costs))

    def __repr__(self):
        res = 'self contrastive + regret'
        if self.smoothing_samples > 0:
            res += f' (ns={self.smoothing_samples}, std={self.smoothing_std})'
        return res


class SPOPlusLoss:
    def __init__(self, smoothing_samples=0, smoothing_std=0.05, seed=42, alpha=2):
        self.smoothing_samples = smoothing_samples
        self.smoothing_std = smoothing_std
        if smoothing_samples > 0:
            self.noise = normal_sample_(mean=0, std=smoothing_std, size=(smoothing_samples, 1, 2), seed=seed)
        else:
            self.noise = np.zeros((1, 1, 2))
        self.alpha = alpha

    def __call__(self, c_star, c_hat, y_star):
        c_hat = c_hat + self.noise
        c_spo = self.alpha * c_hat - c_star
        y_spo = get_decisions_(c_spo)
        best_spo_costs = np.sum(c_spo * y_spo, axis=-1)
        spo_best_costs = np.sum(c_spo * y_star, axis=-1)
        return np.mean(spo_best_costs - best_spo_costs)


        best_est_costs = np.sum(c_hat * y_hat, axis=-1)
        est_best_costs = np.sum(c_hat * y_star, axis=-1)
        return np.mean(est_best_costs - best_est_costs)

    def __repr__(self):
        res = 'SPO+'
        if self.alpha != 2:
            res += f' with alpha={self.alpha}'
        if self.smoothing_samples > 0:
            res += f' (ns={self.smoothing_samples}, std={self.smoothing_std})'
        return res


def draw_loss_landscape(losses,
                        w_min=-0.25,
                        w_max=1.5,
                        w_nvals=1000,
                        x_mean=0.54,
                        x_std=0.2,
                        batch_size=1,
                        model=1,
                        seed=42,
                        figsize=(14, 4)):
    # Draw samples
    x = normal_sample_(mean=x_mean, std=x_std, size=batch_size, seed=seed)
    x = x.reshape(1, -1)
    # Define w values
    w_vals = np.linspace(w_min, w_max, w_nvals)
    # Obtain true costs
    c_star = eval_ground_truth_(x)
    # Obtain the true optimal decisions
    y_star = get_decisions_(c_star)
    # Consider all ws
    lvals_per_w = []
    for w in w_vals:
        # Obtain estimated costs
        c_hat = eval_predictions_(x, w, model=model)
        # Prepare a data structure with all the loss values
        loss_values = [l(c_star, c_hat, y_star) for l in losses]
        # Store the loss vector
        lvals_per_w.append(loss_values)
    # Convert the loss data structure into an array
    lvals_per_w = np.array(lvals_per_w)
    # Draw a figure
    plt.figure(figsize=figsize)
    for i, l in enumerate(losses):
        plt.plot(w_vals, lvals_per_w[:, i], label=str(l))
    plt.title(f'number of examples: {batch_size}')
    plt.xlabel(r'$\theta$')
    plt.legend()
    plt.grid(':')
    plt.show()



def draw_solution_values_(y, w_min, w_max, x_min, x_max, title=None, nticks=5):
    plt.imshow(y, cmap='Greys_r', origin='lower', aspect='auto')
    plt.xticks(ticks=np.linspace(0, y.shape[1]-1, nticks),
               labels=[f'{v:.1f}' for v in np.linspace(x_min, x_max, nticks)])
    plt.yticks(ticks=np.linspace(0, y.shape[0]-1, nticks),
               labels=[f'{v:.1f}' for v in np.linspace(w_min, w_max, nticks)])
    plt.xlabel('x')
    plt.ylabel('w')
    plt.title(title)


def draw_solution_landscape(ytype='y_star',
                        w_min=-1.5,
                        w_max=1.5,
                        w_nvals=100,
                        x_min=0,
                        x_max=1,
                        x_nvals=100,
                        smoothing_samples=0,
                        smoothing_std=0.05,
                        smoothing_seed=42,
                        model=1,
                        figsize=(14, 4)):
    # Stochastic smoothing has no impact on the ground truth solution
    if ytype == 'y_star' and smoothing_samples > 0:
        print('WARNING: stochastic smoothing has no impact on the ground truth solution and will be ignored')
    # Prepare a result data structure
    y0_vals, y1_vals = [], []
    for w in np.linspace(w_min, w_max, w_nvals):
        x = np.linspace(x_min, x_max, x_nvals)
        x = x.reshape(1, -1)
        # Define noise
        if ytype != 'y_star' and smoothing_samples > 0:
            noise = normal_sample_(mean=0, std=smoothing_std,
                                        size=(smoothing_samples, 1, 2),
                                        seed=smoothing_seed)
        else:
            noise = np.zeros((1, 1, 2))
        # Compute the solution
        if ytype == 'y_star':
            c_star = eval_ground_truth_(x)
            y = get_decisions_(c_star)
        elif ytype == 'y_hat':
            c_hat = eval_predictions_(x, w, model=model) + noise
            y = get_decisions_(c_hat)
        elif ytype == 'y_spo':
            c_star = eval_ground_truth_(x)
            c_hat = eval_predictions_(x, w) + noise
            y = get_decisions_(2*c_hat - c_star)
        # Append solution values
        y0_vals.append(np.mean(y[:, :, 0], axis=0))
        y1_vals.append(np.mean(y[:, :, 1], axis=0))
    # Convert to a numpy array
    y0_vals = np.array(y0_vals)
    y1_vals = np.array(y1_vals)
    # Define titles
    if ytype == 'y_star':
        y0_title, y1_title = r'$y^\star_0$', r'$y^\star_1$'
    elif ytype == 'y_hat':
        y0_title, y1_title = r'$\hat{y}_0$', r'$\hat{y}_1$'
    elif ytype == 'y_spo':
        y0_title, y1_title = r'$\hat{y}^{spo}_0$', r'$\hat{y}^{spo}_1$'
    # Add info about smoothing
    if ytype != 'y_star' and smoothing_samples > 0:
        smoothing_info = f' (ns={smoothing_samples}, std={smoothing_std})'
        y0_title += smoothing_info
        y1_title += smoothing_info
    # Draw the solution
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    draw_solution_values_(y0_vals, title=y0_title,
                          w_min=w_min, w_max=w_max, x_min=x_min, x_max=x_max)
    plt.subplot(1, 2, 2)
    draw_solution_values_(y1_vals, title=y1_title,
                          w_min=w_min, w_max=w_max, x_min=x_min, x_max=x_max)
    plt.show()


# ==============================================================================
# DFL
# ==============================================================================

def generate_costs(nsamples, nitems, noise_scale=0,
                   seed=None, sampling_seed=None,
                   nsamples_per_point=1,
                   noise_type='normal', noise_scale_type='absolute'):
    assert(noise_scale >= 0)
    assert(nsamples_per_point > 0)
    assert(noise_type in ('normal', 'rayleigh'))
    assert(noise_scale_type in ('absolute', 'relative'))
    # Seed the RNG
    np.random.seed(seed)
    # Generate costs
    speed = np.random.choice([-14, -11, 11, 14], size=nitems)
    scale = 1.3 + 1.3 * np.random.rand(nitems)
    base = 1 * np.random.rand(nitems) / scale
    offset = -0.75 + 0.5 * np.random.rand(nitems)

    # Generate input
    if sampling_seed is not None:
        np.random.seed(sampling_seed)
    x = np.random.rand(nsamples)
    x = np.repeat(x, nsamples_per_point)

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
    if noise_scale > 0:
        for i in range(nitems):
            # Define the noise scale
            if noise_scale_type == 'absolute':
                noise_scale_vals = noise_scale * res[f'C{i}']**0
            elif noise_scale_type == 'relative':
                noise_scale_vals = noise_scale * res[f'C{i}']
            # Define the noise distribution
            if noise_type == 'normal':
                noise_dist = stats.norm(scale=noise_scale_vals)
            elif noise_type == 'rayleigh':
                # noise_dist = stats.expon(scale=noise_scale_vals)
                noise_dist = stats.rayleigh(scale=noise_scale_vals)
            r_mean = noise_dist.mean()
            # pnoise = noise * np.random.randn(nsamples)
            # res[f'C{i}'] = res[f'C{i}'] + pnoise
            pnoise = noise_dist.rvs()
            res[f'C{i}'] = res[f'C{i}'] + pnoise - r_mean
    # Reindex
    res.set_index('x', inplace=True)
    # Sort by index
    res.sort_index(inplace=True)
    # Normalize
    vmin, vmax = res.min().min(), res.max().max()
    res = (res - vmin) / (vmax - vmin)

    # Return results
    return res


def plot_df_cols(data, scatter=False, figsize=None, legend=True, title=None):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    x = data.index
    plt.xlabel(data.index.name)
    # Plot all columns
    for cname in data.columns:
        y = data[cname]
        plt.plot(x, y, label=cname,
                 linestyle='-' if not scatter else '',
                 marker=None if not scatter else '.',
                 alpha=1 if not scatter else 0.3)
    # Add legend
    if legend and len(data.columns) <= 10:
        plt.legend(loc='best')
    plt.grid(':')
    # Add a title
    plt.title(title)
    # Make it compact
    plt.tight_layout()
    # Show
    plt.show()


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


def plot_nn_model(model, show_layer_names=True, show_layer_activations=True, show_shapes=True):
    return keras.utils.plot_model(model, show_shapes=show_shapes,
            show_layer_names=show_layer_names, rankdir='LR',
            show_layer_activations=show_layer_activations)


def train_nn_model(model, X, y, loss,
        verbose=0, patience=10,
        validation_split=0.0,
        save_weights=False,
        load_weights=False,
        **fit_params):
    # Try to load the weights, if requested
    if load_weights:
        history = load_ml_model_weights_and_history(model, model.name)
        return history
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
    # Save the model, if requested
    if save_weights:
        save_ml_model_weights_and_history(model, model.name, history)
    return history


def plot_training_history(history=None,
        figsize=None, print_final_scores=True, excluded_metrics=[]):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        if metric not in excluded_metrics:
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


def save_ml_model_weights_and_history(model, name, history):
    # Save names
    wgt_fname = os.path.join('data', f'{name}.h5')
    model.save_weights(wgt_fname)
    # Save history
    hst_fname = os.path.join('data', f'{name}.history')
    with open(hst_fname, 'wb') as fp:
        pickle.dump(history, fp)


def save_ml_model(model, name):
    save_ml_model_weights(model, name)
    mdl_fname = os.path.join('data', f'{name}.json')
    with open(mdl_fname, 'w') as f:
        f.write(model.to_json())


def load_ml_model_weights_and_history(model, name):
    # Load weights
    wgt_fname = os.path.join('data', f'{name}.h5')
    if os.path.exists(wgt_fname):
        model.load_weights(wgt_fname)
    # Load history
    hst_fname = os.path.join('data', f'{name}.history')
    with open(hst_fname, 'rb') as fp:
        history = pickle.load(fp)
    return history


def load_ml_model(name):
    mdl_fname = os.path.join('data', f'{name}.json')
    if not os.path.exists(mdl_fname):
        return None
    with open(mdl_fname) as f:
        model = keras.models.model_from_json(f.read())
        load_ml_model_weights(model, name)
        return model


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


def print_ml_metrics(model, X, y, label=''):
    r2, mae, rmse = get_ml_metrics(model, X, y)
    if len(label) > 0:
        label = f' ({label})'
    print(f'R2: {r2:.2f}, MAE: {mae:.2}, RMSE: {rmse:.2f}{label}')


def generate_problem(nitems, rel_req, seed=None, surrogate=False):
    # Seed the RNG
    np.random.seed(seed)
    # Generate the item values
    values = 1 + 0.4*np.random.rand(nitems)
    # Generate the requirement
    req = rel_req * np.sum(values)
    # Return the results
    if not surrogate:
        return ProductionProblem(values, req)
    else:
        return ProductionProblemSurrogate(values, req)


def generate_2s_problem(nitems, requirement, rel_buffer_cost,
                                seed=None, integer_vars=True):
    # Seed the RNG
    np.random.seed(seed)
    # Generate the item costs
    costs = 1 + 0.4*np.random.rand(nitems)
    # Generate the cost of the buffer product
    buffer_cost = rel_buffer_cost * np.mean(costs)
    # Return the results
    return ProductionProblem2Stage(costs, requirement, buffer_cost,integer_vars=integer_vars)


class ProductionProblem(object):
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
            # Attach the computed cost
            res.append(slv.Objective().Value())
            # Attach the solution time
            res.append(slv.wall_time()/1000.0)
        else:
            # TODO I am not handling the unbounded case
            # It should never arise in the first place
            if status == slv.INFEASIBLE:
                res = [None, True, None, slv.wall_time()/1000.0]
            else:
                res = [None, False, None, slv.wall_time()/1000.0]

        # Print the solution, if requested
        if print_solution:
            self._print_sol(res[0], res[1], costs)
        return res

    def _print_sol(self, sol, closed, costs):
        # Obtain indexes of selected items
        idx = [i for i in range(len(sol)) if sol[i] > 0]
        # Print selected items with values and costs
        s = ', '.join(f'{i}' for i in idx)
        print('Selected items:', s)
        s = f'Cost: {sum(costs):.2f}, '
        s += f'Value: {sum(self.values):.2f}, '
        s += f'Requirement: {self.requirement:.2f}, '
        s += f'Closed: {closed}'
        print(s)

    def __repr__(self):
        return f'ProductionProblem(values={self.values}, requirement={self.requirement})'


class ProductionProblem2Stage(object):
    def __init__(self, costs, requirement, buffer_cost, integer_vars=True):
        """TODO: to be defined. """
        # Store the problem configuration
        self.costs = costs
        self.requirement = requirement
        self.buffer_cost = buffer_cost
        self.integer_vars = integer_vars

    def solve(self, values, fixed_fsd=None, tlim=None, print_solution=False):
        # Handle the case where values is a single vector
        try:
            values[0][0]
        except:
            values = [values]
        # Quick access to some useful fields
        costs = self.costs
        req = self.requirement
        req_ub = int(np.ceil(req + np.max(np.sum(np.maximum(values, 0), axis=1))))
        bcst = self.buffer_cost
        nv = len(costs)
        ns = len(values) # number of scenario
        # Build the solver
        slv = pywraplp.Solver.CreateSolver('CBC')
        # Build the first variables
        if self.integer_vars:
            x = [slv.IntVar(0, 1, f'x_{i}') for i in range(nv)]
        else:
            x = [slv.NumVar(0, 1, f'x_{i}') for i in range(nv)]
        # Apply first-stage decisions (if specified)
        if fixed_fsd is not None:
            for i in range(nv):
                slv.Add(x[i] == fixed_fsd[i])
        # Build the second stage variables
        if self.integer_vars:
            y = [slv.IntVar(0, req_ub, f'y_{k}') for k in range(ns)]
        else:
            y = [slv.NumVar(0, req_ub, f'y_{k}') for k in range(ns)]
        # Build the requirement constraint (for every scenario)
        for k in range(ns):
            slv.Add(sum(values[k][i] * x[i] for i in range(nv)) + y[k] >= req)
        # Build the objective
        slv.Minimize(sum(costs[i] * x[i] for i in range(nv)) + bcst/ns * sum(y[k] for k in range(ns)))
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
            # Attach the computed cost
            res.append(slv.Objective().Value())
            # Attach the solution time
            res.append(slv.wall_time()/1000.0)
        else:
            # TODO I am not handling the unbounded case
            # It should never arise in the first place
            if status == slv.INFEASIBLE:
                res = [None, True, None, slv.wall_time()/1000.0]
            else:
                res = [None, False, None, slv.wall_time()/1000.0]
        # Print the solution, if requested
        if print_solution:
            self._print_sol(res[0], res[1], res[2],
                      values, [y[k].solution_value() for k in range(ns)])
        return res

    def rel_evpf(self, sol, true_values, tlim=None):
        # Compute the expected cost of the solution
        sol, _, obj, _  = self.solve(true_values, fixed_fsd=sol, tlim=tlim)
        # For every scenario, compute the perfect information solution
        pf_objs = []
        for tv in true_values:
            _, _, pf_obj, _  = self.solve([tv], tlim=tlim)
            pf_objs.append(pf_obj)
        pf_objs = np.array(pf_objs)
        # Compute the average regret w.r.t. the perfect information solutions
        return np.mean((obj - pf_objs) / pf_objs)

    def rel_regret(self, sol, true_values, tlim=None):
        # Compute the expected cost of the solution
        sol, _, obj, _  = self.solve(true_values, fixed_fsd=sol, tlim=tlim)
        # Compute the best possibe solution (assuming a perfect sample)
        tsol, _, tobj, _  = self.solve(true_values, tlim=tlim)
        # Compute the average regret w.r.t. the perfect information solutions
        return (obj - tobj) / np.abs(tobj)

    def _print_sol(self, sol, closed, obj, values, y_vals):
        # Obtain indexes of selected items
        idx = [i for i in range(len(sol)) if sol[i] > 0]
        # Print selected items with values and costs
        s = ', '.join(f'{i}' for i in idx)
        print('Selected items:', s)
        s = f'Cost: {obj:.2f}, '
        s += f'Recourse Actions: {y_vals}, '
        s += f'Requirement: {self.requirement:.2f}, '
        s += f'Closed: {closed}'
        print(s)

    def __repr__(self):
        return f'ProductionProblem2Stage(costs={self.costs}, requirement={self.requirement}, buffer_cost={self.buffer_cost})'



def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


class AckleyFunction(object):

    def __init__(self):
        self.n = 2 # number of dimensions

    def _eval_true(self, x):
        x = 5 * (x-0.1) # rescale and translate
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(np.square(x), axis=1))) \
                - np.exp(0.5*np.sum(np.cos(2 * np.pi * x),axis=1)) + np.exp(0) + 20

    def _eval_surrogate(self, x, values):
        n = self.n
        pQ = values[:n**2].reshape(n, n)
        Q = np.matmul(pQ.T, pQ)
        xs = values[n**2:n**2+n].reshape((1, n))
        return np.sum(np.matmul(x - xs, Q) * (x - xs), axis=1)

    def solve(self, values=None, fixed_fsd=None, tlim=None, print_solution=False,
              grid_samples=100):
        # Number of dimensions
        nv = 2
        # assert(values is None or len(values) == nv)
        # Generate the solutions to be tested
        sols = cartesian_product([np.linspace(0, 1, grid_samples) for i in range(nv)])
        # Compute either the true or the surrogate cost
        if values is None:
            cost = self._eval_true(sols)
        else:
            values = np.array(values)
            cost = self._eval_surrogate(sols, values)
        # Find the solution with minimal cost
        best_idx = np.argmin(cost)
        best_sol = sols[best_idx]
        best_cost = cost[best_idx]
        # Return results

        print(best_sol, best_cost)

    # def _print_sol(self, sol, closed, obj, values, y_vals):
    #     # Obtain indexes of selected items
    #     idx = [i for i in range(len(sol)) if sol[i] > 0]
    #     # Print selected items with values and costs
    #     s = ', '.join(f'{i}' for i in idx)
    #     print('Selected items:', s)
    #     s = f'Cost: {obj:.2f}, '
    #     s += f'Recourse Actions: {y_vals}, '
    #     s += f'Requirement: {self.requirement:.2f}, '
    #     s += f'Closed: {closed}'
    #     print(s)

    def __repr__(self):
        return f'ProductionProblem2Stage(costs={self.costs}, requirement={self.requirement}, buffer_cost={self.buffer_cost})'




def compute_evpf_2s(problem, predictor, data_ts, tlim=None):
    # Obtain all inputs
    pred_in = np.unique(data_ts.index)
    # Compute all predictions
    pred_values = predictor.predict(pred_in, verbose=0)
    # Compute the EVPF for every input
    evpfs = []
    for pin, pvals in zip(pred_in, pred_values):
        # Compute the solution
        sol, closed, obj, wtime = problem.solve([pvals], tlim=tlim)
        # Obtain the true value
        true_values = data_ts.loc[pin].values
        # Compute the EVPF for this solution
        evpf = problem.rel_evpf(sol, true_values, tlim=tlim)
        evpfs.append(evpf)
    # Return results
    return np.array(evpfs)


def compute_regret_2s(problem, predictor, data_ts, tlim=None):
    # Obtain all inputs
    pred_in = np.unique(data_ts.index)
    # Compute all predictions
    pred_values = predictor.predict(pred_in, verbose=0)
    # Compute the EVPF for every input
    regrets = []
    for pin, pvals in zip(pred_in, pred_values):
        # Compute the solution
        sol, closed, obj, wtime = problem.solve([pvals], tlim=tlim)
        # Obtain the true value
        true_values = data_ts.loc[pin].values
        # Compute the EVPF for this solution
        regret = problem.rel_regret(sol, true_values, tlim=tlim)
        regrets.append(regret)
    # Return results
    return np.array(regrets)


def compute_regret(problem, predictor, pred_in, true_costs, tlim=None):
    # Obtain all predictions
    costs = predictor.predict(pred_in, verbose=0)
    # Compute all solutions
    sols = []
    for c in costs:
        sol, _, _, _  = problem.solve(c, tlim=tlim)
        sols.append(sol)
    sols = np.array(sols)
    # Compute the true solutions
    tsols = []
    for c in true_costs:
        sol, _, _, _ = problem.solve(c, tlim=tlim)
        tsols.append(sol)
    tsols = np.array(tsols)
    # Compute true costs
    costs_with_predictions = np.sum(true_costs * sols, axis=1)
    costs_with_true_solutions = np.sum(true_costs * tsols, axis=1)
    # Compute regret
    regret = (costs_with_predictions - costs_with_true_solutions) / np.abs(costs_with_true_solutions)
    # Return true costs
    return regret



def plot_histogram(data, label=None, bins=20, figsize=None,
        data2=None, label2=None, print_mean=False, title=None):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    if data2 is None:
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
    # Add a title
    plt.title(title)
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
        output_activation='linear', loss_type='scr',
        sfge=False, sfge_sigma_init=1, sfge_sigma_trainable=False,
        surrogate=False, standardize_loss=False, name=None):
    assert(not sfge or recompute_chance==1)
    assert(not sfge or loss_type in ('cost', 'regret', 'scr'))
    assert(sfge or loss_type in ('scr', 'spo', 'sc'))
    assert(sfge_sigma_init > 0)
    assert(not surrogate or (sfge and loss_type == 'cost'))
    # Build all layers
    nnin = keras.Input(input_size)
    nnout = nnin
    for h in hidden:
        nnout = layers.Dense(h, activation='relu')(nnout)
    if output_activation != 'linear_normalized' or output_size == 1:
        nnout = layers.Dense(output_size, activation=output_activation)(nnout)
    else:
        h1_out = layers.Dense(output_size-1, activation='linear')(nnout)
        h2_out = tf.reshape(1 - tf.reduce_sum(h1_out, axis=1), (-1, 1))
        nnout = tf.concat([h1_out, h2_out], axis=1)
    # Build the model
    if not sfge:
        model = DFLModel(problem, tlim=tlim, recompute_chance=recompute_chance,
                    inputs=nnin, outputs=nnout, loss_type=loss_type,
                    name=name)
    else:
        model = SFGEModel(problem, tlim=tlim,
                    inputs=nnin, outputs=nnout, loss_type=loss_type,
                    sigma_init=sfge_sigma_init, sigma_trainable=sfge_sigma_trainable,
                    surrogate=surrogate, standardize_loss=standardize_loss, name=name)
    return model


def train_dfl_model(model, X, y, tlim=None,
        epochs=20, verbose=0, patience=10, batch_size=32,
        validation_split=0.2, optimizer='Adam',
        save_weights=False,
        load_weights=False,
        warm_start_pfl=None,
        **params):
    # Try to load the weights, if requested
    if load_weights:
        history = load_ml_model_weights(model, model.name)
        return history
    # Attempt a warm start
    if warm_start_pfl is not None:
        transfer_weights_to_dfl_model(warm_start_pfl, model)
    # Compile and train
    model.compile(optimizer=optimizer, run_eagerly=True)
    if validation_split > 0:
        cb = [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    else:
        cb = None
    # Start training
    history = model.fit(X, y, validation_split=validation_split,
                     callbacks=cb, batch_size=batch_size,
                     epochs=epochs, verbose=verbose, **params)
    # Save the model, if requested
    if save_weights:
        save_ml_model_weights_and_history(model, model.name, history)
    return history



class DFLModel(keras.Model):
    def __init__(self, prb, tlim=None, recompute_chance=1,
                 loss_type='spo',
                 **params):
        super(DFLModel, self).__init__(**params)
        assert(loss_type in ('scr', 'spo', 'sc'))
        # Store configuration parameters
        self.prb = prb
        self.tlim = tlim
        self.recompute_chance = recompute_chance
        self.loss_type = loss_type

        # Build metrics
        self.metric_loss = keras.metrics.Mean(name="loss")
        # self.metric_regret = keras.metrics.Mean(name="regret")
        # self.metric_mae = keras.metrics.MeanAbsoluteError(name="mae")
        # Prepare a field for the solutions
        self.sol_store = None
        self.tsol_index = None


    def fit(self, X, y, **kwargs):
        # Precompute all solutions for the true costs
        self.sol_store = []
        self.tsol_index = {}
        for x, c in zip(X.astype('float32'), y):
            sol, closed, obj, _ = self.prb.solve(c, tlim=self.tlim)
            self.sol_store.append(sol)
            self.tsol_index[x] = len(self.sol_store)-1
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
            # Define the costs to be used for computing the solutions
            costs_iter = costs
            # Compute SPO costs, if needed
            if self.loss_type == 'spo':
                spo_costs = 2*costs - costs_true
                costs_iter = spo_costs
            # Solve all optimization problems
            sols, tsols = [], []
            for xv, c, tc in zip(x.numpy(), costs_iter.numpy(), costs_true.numpy()):
                # Decide whether to recompute the solution
                if np.random.rand() < self.recompute_chance:
                    sol, closed, _, _ = prb.solve(c, tlim=self.tlim)
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
                # tsol = self._find_best(tc)
                tsol = self.sol_store[self.tsol_index[xv]]
                tsols.append(tsol)
            # Convert solutions to numpy arrays
            sols = np.array(sols)
            tsols = np.array(tsols)
            # Compute the loss
            if self.loss_type == 'scr':
                # Compute the cost difference
                cdiff = costs - costs_true
                # Compute the solution difference
                sdiff = tsols - sols
                # Compute the loss terms
                loss_terms = tf.reduce_sum(cdiff * sdiff, axis=1)
            elif self.loss_type == 'spo':
                loss_terms = tf.reduce_sum(spo_costs * (tsols - sols), axis=1)
            elif self.loss_type == 'sc':
                loss_terms = tf.reduce_sum(costs * (tsols - sols), axis=1)
            # Compute the mean loss
            loss = tf.reduce_mean(loss_terms)

        # Perform a gradient descent step
        tr_vars = self.trainable_variables
        gradients = tape.gradient(loss, tr_vars)
        self.optimizer.apply_gradients(zip(gradients, tr_vars))

        # Update main metrics
        self.metric_loss.update_state(loss)
        # Update compiled metrics
        self.compiled_metrics.update_state(costs_true, costs)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.metric_loss]




class SFGEModel(keras.Model):
    def __init__(self, prb, tlim=None,
                 loss_type='cost',
                 sigma_init=1,
                 sigma_trainable=False,
                 surrogate=False,
                 standardize_loss=False,
                 **params):
        super(SFGEModel, self).__init__(**params)
        assert(loss_type in ('cost', 'regret', 'scr'))
        assert(not surrogate or loss_type in ('cost'))
        # Store configuration parameters
        self.prb = prb
        self.tlim = tlim
        self.loss_type = loss_type
        self.surrogate = surrogate
        self.standardize_loss = standardize_loss

        # Prepare objects to handle Score Function Gradient Estimation
        self.log_sigma = tf.Variable(np.log(sigma_init), dtype=np.float32,
                                     name='log_sigma', trainable=sigma_trainable)

        # Build metrics
        self.metric_sigma = keras.metrics.Mean(name="sigma")
        self.metric_sample_cost = keras.metrics.Mean(name="sample_cost")
        self.metric_loss = keras.metrics.Mean(name="loss")

        # Prepare a field for the solutions
        self.sol_store = None
        self.tsol_index = None
        self.tsol_obj = None

    def fit(self, X, y, **kwargs):
        # Precompute all solutions for the true costs
        # NOTE this is not triggered in case a surrogate is used
        self.sol_store = []
        self.tsol_index = {}
        self.tsol_obj = {}
        if not self.surrogate:
            for x, p in zip(X.astype('float32'), y):
                sol, closed, obj, _ = self.prb.solve(p, tlim=self.tlim)
                self.sol_store.append(sol)
                self.tsol_index[x] = len(self.sol_store)-1
                self.tsol_obj[x] = obj
            self.sol_store = np.array(self.sol_store)
        # Call the normal fit method
        return super(SFGEModel, self).fit(X, y, **kwargs)

    # def _find_best(self, costs):
    #     tc = np.dot(self.sol_store, costs)
    #     best_idx = np.argmin(tc)
    #     best = self.sol_store[best_idx]
    #     return best

    def train_step(self, data):
        # Unpack the data
        x, params_true = data
        # Quick access to some useful fields
        prb = self.prb
        tlim = self.tlim

        # Loss computation
        with tf.GradientTape() as tape:
            # Obtain the prediction mean
            params = self(x, training=True)
            # Sample
            sigma = tf.math.exp(self.log_sigma)
            dist = tfp.distributions.Normal(loc=params, scale=sigma)
            sampler = tfp.distributions.Sample(dist, sample_shape=1)
            sample_params = sampler.sample()[:, :, 0].numpy()

            # Prepare data structures to store loss term info
            objs, pobjs, tobjs, tpobjs = [], [], [], []

            # Compute sample log probabilities
            sample_logprobs = tf.reduce_sum(sampler.distribution.log_prob(sample_params), axis=1)
            # Solve all optimization problems
            for xv, p, tp in zip(x.numpy(), sample_params, params_true.numpy()):
                # Compute the optimal solution w.r.t. the estimated parameters
                sol, closed, pobj, _ = prb.solve(p, tlim=self.tlim)
                pobjs.append(pobj)
                # Compute the actual objective for this solution
                if not self.surrogate:
                    _, closed, obj, _ = prb.solve(tp, fixed_fsd=sol, tlim=self.tlim)
                else:
                    obj = prb.eval(sol, tp)
                objs.append(obj)
                # Find the best solution with the true costs
                if self.loss_type in ('regret', 'scr'):
                    tobj = self.tsol_obj[xv]
                    tobjs.append(tobj)
                # Compute the predicted objective for the true solution
                if self.loss_type in ('scr'):
                    tsol = self.sol_store[self.tsol_index[xv]]
                    _, closed, tpobj, _ = prb.solve(p, fixed_fsd=tsol, tlim=self.tlim)
                    tpobjs.append(tpobj)

            # Convert objective to numpy arrays
            objs = np.array(objs)
            tobjs = np.array(tobjs)
            pobjs = np.array(pobjs)
            tpobjs = np.array(tpobjs)
            # Compute the loss
            if self.loss_type == 'cost':
                loss_terms = objs
            elif self.loss_type == 'regret':
                loss_terms = objs - tobjs
            elif self.loss_type == 'scr':
                loss_terms = (objs - tobj) + (tpobjs - pobjs)
            # Apply standardization
            if self.standardize_loss:
                loss_term_mean = tf.reduce_mean(loss_terms).numpy()
                loss_term_std = tf.math.reduce_std(loss_terms).numpy()
                loss_terms = (loss_terms - loss_term_mean) / loss_term_std
            # Compute the mean loss
            loss = tf.reduce_mean(loss_terms * sample_logprobs)

        # Perform a gradient descent step
        tr_vars = self.trainable_variables
        gradients = tape.gradient(loss, tr_vars)
        gvpairs = zip(gradients, tr_vars)
        self.optimizer.apply_gradients(gvpairs)

        # Update main metrics
        self.metric_loss.update_state(loss)
        self.metric_sigma.update_state(tf.math.exp(self.log_sigma))
        self.metric_sample_cost.update_state(tf.reduce_mean(objs))
        # regrets = tf.reduce_sum((sols - tsols) * costs_true, axis=1)
        # mean_regret = tf.reduce_mean(regrets)
        # self.metric_regret.update_state(mean_regret)
        # self.metric_mae.update_state(costs_true, costs)
        # Update compiled metrics
        self.compiled_metrics.update_state(params_true, params)
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
        return [self.metric_loss, self.metric_sigma, self.metric_sample_cost]


def transfer_weights_to_dfl_model(source, dest):
    source_weights = source.get_weights()
    dest_weights = dest.get_weights()
    transfer_weights = source_weights[:len(source_weights)] + dest_weights[len(source_weights):]
    dest.set_weights(transfer_weights)



class ProductionProblemSurrogate(object):
    """Docstring for MarketProblem. """

    def __init__(self, values, requirement):
        """TODO: to be defined. """
        # Store the problem configuration
        self.values = values
        self.requirement = requirement

    def eval(self, sol, costs):
        # terms = np.exp(costs * sol)
        # terms = np.log(0.01 + costs * sol)
        terms = np.sin(2 * np.pi * costs * sol)
        if len(terms.shape) > 1:
            return np.sum(terms, axis=1)
        else:
            return np.sum(terms)

    def solve_exact(self, costs, samples_per_item=11):



        # Quick access to some useful fields
        values = self.values
        req = self.requirement
        nv = len(values)
        # Generate the solutions
        sols = cartesian_product([np.linspace(0, 1, samples_per_item) for i in range(nv)])
        # Filter infeasible solutions
        mask = np.sum(np.array(values).reshape(1, -1) * sols, axis=1) >= req
        sols = sols[mask]
        # Compute all actual objectives
        objs = self.eval(sols, costs)
        # Pick and return the best solution
        best_idx = np.argmin(objs)
        best_sol = sols[best_idx]
        best_obj = objs[best_idx]
        return best_sol, best_obj


    # def solve(self, costs, samples_per_item=11, tlim=None):
    #     # Quick access to some useful fields
    #     values = self.values
    #     req = self.requirement
    #     nv = len(values)
    #     # Generate the solutions
    #     sols = cartesian_product([np.linspace(0, 1, samples_per_item) for i in range(nv)])
    #     # Filter infeasible solutions
    #     mask = np.sum(np.array(values).reshape(1, -1) * sols, axis=1) >= req
    #     sols = sols[mask]
    #     # Compute all actual objectives
    #     objs = np.sum(costs * sols, axis=1)
    #     # Pick and return the best solution
    #     best_idx = np.argmin(objs)
    #     best_sol = sols[best_idx]
    #     best_obj = objs[best_idx]
    #     return best_sol, True, best_obj, None

    def solve(self, costs, tlim=None):
        # Quick access to some useful fields
        values = self.values
        req = self.requirement
        nv = len(values)
        # Build the solver
        slv = pywraplp.Solver.CreateSolver('CBC')
        # Build the variables
        x = [slv.NumVar(0, 1, f'x_{i}') for i in range(nv)]
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
            # Attach the computed cost
            res.append(slv.Objective().Value())
            # Attach the solution time
            res.append(slv.wall_time()/1000.0)
        else:
            # TODO I am not handling the unbounded case
            # It should never arise in the first place
            if status == slv.INFEASIBLE:
                res = [None, True, None, slv.wall_time()/1000.0]
            else:
                res = [None, False, None, slv.wall_time()/1000.0]

        # # Print the solution, if requested
        # if print_solution:
        #     self._print_sol(res[0], res[1], costs)
        return res

    def __repr__(self):
        return f'ProductionProblemSurrogate(values={self.values}, requirement={self.requirement})'


def compute_regret_surrogate(problem, predictor, data, tlim=None,
                             samples_per_item=11, cost_only=False):
    # Obtain all inputs
    pred_in = np.unique(data.index)
    # Compute all predictions
    pred_costs = predictor.predict(pred_in, verbose=0)
    # Compute the EVPF for every input
    actual_objs, best_objs = [], []
    for pin, pcosts in zip(pred_in, pred_costs):
        # Compute the solution
        sol, closed, obj, wtime = problem.solve(pcosts, tlim=tlim)
        # Obtain the true costs
        true_costs = data.loc[pin].values
        # Compute the true cost for this solution
        actual_obj = problem.eval(sol, true_costs)
        actual_objs.append(actual_obj)
        # Compute the best possible cost for this solution
        if not cost_only:
            best_sol, best_obj = problem.solve_exact(true_costs, samples_per_item)
            best_objs.append(best_obj)
    # Return results
    actual_objs = np.array(actual_objs)
    if not cost_only:
        best_objs = np.array(best_objs)
        return (actual_objs - best_objs) / np.abs(best_objs)
    else:
        return actual_objs



# def column_histograms(data, title=None):
#     # Normalize all data
#     ndata = (data - data.mean()) / data.std()
#     # Compute histograms


#     plt.imshow(y, cmap='Greys_r', origin='lower', aspect='auto')
#     plt.xticks(ticks=np.linspace(0, y.shape[1]-1, nticks),
#                labels=[f'{v:.1f}' for v in np.linspace(x_min, x_max, nticks)])
#     plt.yticks(ticks=np.linspace(0, y.shape[0]-1, nticks),
#                labels=[f'{v:.1f}' for v in np.linspace(w_min, w_max, nticks)])
#     plt.xlabel('x')
#     plt.ylabel('w')
#     plt.title(title)
