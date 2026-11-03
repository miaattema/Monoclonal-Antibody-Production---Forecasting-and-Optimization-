# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

def calculate_average(values):
    return sum(values) / len(values)


import sys

import numpy as np


def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0:
        progress = c_iter / n_iters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")


def generate_x_lag_cur_ar(p, batch_ids, df, feature, col_batch="Batch ID", col_time="Time [h]"):

    x_lag = []
    x_cur = []
    t_cur = []

    for batch_id in batch_ids:

        mask = df.loc[:, col_batch] == batch_id
        data_batch = df.loc[mask, feature].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()

        current_values = data_batch[p:]
        current_times = t_batch[p:]

        lagged_values = []
        for i in range(1, p + 1):
            lagged_values.append(data_batch[p - i:-i])

        lagged_values = np.stack(lagged_values, axis=1)

        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)

    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)

    return x_lag, x_cur, t_cur


def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch="Batch ID", col_time="Time [h]"):

    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    for batch_id in batch_ids:
        mask = df.loc[:, col_batch] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()

        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]

        lagged_values = []
        for i in range(1, p + 1):
            lagged_values.append(x_batch[p - i:-i])

        lagged_values = np.concatenate(lagged_values, axis=1)

        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)

    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur

# ============================================================================
# FORECASTING FUNCTION
# ============================================================================

def simulate_trajectories(initial_glucose, initial_glutamine, initial_vcd,
                          model, residuals, p, features,
                          n_simulations, rng, n_steps):

    # Initial conditions
    x_initial = np.array([
        initial_glucose,
        initial_glutamine,
        0,
        0,
        initial_vcd,
        initial_vcd,
        0
    ])

    n_features = len(features)

    # Store forecast simulations
    x_forecasts = np.zeros((n_simulations, n_steps, n_features))

    # Simulate multiple forecasts
    for i_sim in range(n_simulations):

        # Initialize history using the initial conditions
        history = [x_initial.copy() for _ in range(p)]

        for i_step in range(n_steps):

            # Extract lagged values (most recent first)
            x_lag = np.concatenate([
                history[-i] for i in range(1, p + 1)
            ]).reshape(1, -1)

            # Predict the next time point
            x_pred = model.predict(x_lag)[0]

            # Randomly sample residuals for each variable
            white_noise = np.array([
                rng.choice(residuals[:, j])
                for j in range(n_features)
            ])

            # Add residual noise to the prediction
            x_next = x_pred + white_noise

            # Prevent negative concentrations
            x_next = np.clip(x_next, a_min=0, a_max=None)

            # Store the prediction
            x_forecasts[i_sim, i_step, :] = x_next

            # Update history
            history.append(x_next)

    return x_forecasts

