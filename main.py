# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

import os
os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.classes import Batch  # This allows you to import individual Classes from classes.py
from src.constants import *  # This imports all constants from constants.py

ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batch = Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal)
batch.display_info()


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.classes import StandardScalerVAR
from src.functions import generate_x_lag_cur_var, progress_bar
from src.functions import calculate_average




n_simulations = 500
#n_simulations = 5 for testing/ease of running
p_max = 3

prediction_interval = 0.95
alpha = 1 - prediction_interval

rng = np.random.default_rng(0)

# Import the time series dataset; need to make this a for loop to go through all the data series
batch_ids = np.array([36, 108, 133, 142, 183, 187, 206, 215, 260, 349, 373, 383, 396, 430, 530, 547, 571, 591, 615, 648, 657, 661, 693, 725, 771, 810, 813, 832, 925, 1020])
# Read each batch dataset
batch_data = []

for batch_id in batch_ids:
    file_path = os.path.join("datasets", f"batch_{batch_id:04d}.csv")
    df_batch = pd.read_csv(file_path)

    # Add a column identifying which batch the data came from
    df_batch.insert(0, "Batch ID", batch_id)

    batch_data.append(df_batch)

# Combine all batches into one DataFrame
df = pd.concat(batch_data, ignore_index=True)

# Sort by batch and time
df = df.sort_values(["Batch ID", "Time [h]"]).reset_index(drop=True)

# Import information describing the initial conditions of each batch
df_ic = pd.read_csv(os.path.join("datasets", "initial_conditions.csv"))
df_ic = df_ic[df_ic["Batch ID"].isin(batch_ids)].copy()

# Select all process variables to include in the VAR model
features = df.columns[2:]

# Select the variable that will be plotted and evaluated
feature_plot = "mAb [mg/L]"
if feature_plot not in features:
    raise ValueError(f'{feature_plot} is not a column in the dataset.')
idx_fp = list(features).index(feature_plot)

# Extract units for plotting and reporting
units = feature_plot.split(" [")[-1][:-1]

# Organize plots into a 5 x 6 grid (one subplot per batch)
nrows = 5
ncols = 6


print_var_arrays = False
if print_var_arrays:
    x_lag_print, x_cur_print, _, _ = generate_x_lag_cur_var(p=3, batch_ids=batch_ids, df=df, features=features)
    for i in range(x_cur_print.shape[0]):
        print(x_cur_print[i], x_lag_print[i])

n_iters = p_max * batch_ids.shape[0] * n_simulations
c_iter = 0
records = []

# Evaluate VAR models with different numbers of lagged values
for p in range(1, p_max + 1):
    # Create a figure containing one subplot for each set of experimental initial conditions.
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12), dpi=300, layout="constrained",squeeze=False)
    y_min = 0
    y_max = np.max(df.loc[:, feature_plot])
    rmse_forecasts = []

    # Leave-one-batch-out cross-validation
    for batch_index, batch_id_test in enumerate(batch_ids):

        # Training batches contain all batches except the test batch
        batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])

        # Generate the lagged value dataset used to train VAR(p)
        x_lag_train, x_cur_train, _, _ = generate_x_lag_cur_var(p=p, batch_ids=batch_ids_train, df=df,
                                                                features=features)

        # Create and train the model
        model = make_pipeline(StandardScalerVAR(p=p), LinearRegression())
        model.fit(x_lag_train, x_cur_train)

        # Calculate residual errors on the training data.
        # These residuals will later be sampled during forecast simulations to estimate white noise.
        x_cur_train_pred = model.predict(x_lag_train)
        x_cur_train_res = x_cur_train_pred - x_cur_train


        # Find the initial conditions for the test batch
        idx_ic = df_ic.index[df_ic["Batch ID"] == batch_id_test][0]

        initial_glucose = df_ic.at[idx_ic, "Glucose [mM]"]
        initial_glutamine = df_ic.at[idx_ic, "Glutamine [mM]"]
        initial_vcd = df_ic.at[idx_ic, "VCD [cells/mL]"]

        # Construct the initial state containing all process variables
        x_initial = np.array([initial_glucose, initial_glutamine, 0, 0, initial_vcd, initial_vcd, 0])

        # Determine the subplot corresponding to this batch
        i_ax = batch_index // ncols
        j_ax = batch_index % ncols
        ax = axes[i_ax, j_ax]

        # Extract the data for the test batch
        mask = df.loc[:, "Batch ID"] == batch_id_test
        t_test = df.loc[mask, "Time [h]"].to_numpy()
        x_test = df.loc[mask, features].to_numpy()

        # Perform multiple forecast simulations to estimate uncertainty
        x_forecasts = []
        for simulation_id in range(n_simulations):
            # Initialize the forecast using the known initial conditions.
            # Unlike the AR model, each state contains all process variables.
            x_forecast = [x_initial]

            # Generate forecasts recursively.
            # Each newly predicted state becomes available for future predictions.
            while len(x_forecast) < t_test.shape[0]:

                x_lag = x_forecast[::-1][:p].copy()

                # At the beginning of the simulation there may be fewer than p available lagged vectors.
                # If necessary, repeat the most recent state until p lagged vectors are available.
                while len(x_lag) < p:
                    x_lag.append(x_lag[-1])

                # Concatenate all lagged vectors into a single feature vector expected by the VAR model.
                x_lag = np.concatenate(x_lag, axis=0)
                x_lag = np.expand_dims(x_lag, axis=0)

                # Sample residual errors from the training data.
                # This assumes the residuals behave like white noise.
                white_noise = []
                for j in range(x_cur_train_res.shape[1]):
                    white_noise.append(rng.choice(x_cur_train_res[:, j]))
                white_noise = np.array([white_noise])

                # Predict the next process state and add white noise
                x_cur = model.predict(x_lag) + white_noise

                # Prevent negative concentration predictions
                x_cur = np.clip(x_cur[0], a_min=0, a_max=None)

                x_forecast.append(x_cur)

            x_forecast = np.array(x_forecast)

            x_forecasts.append(x_forecast)

            c_iter += 1
            progress_bar(c_iter, n_iters)

        x_forecasts = np.array(x_forecasts)

        # Calculate the median forecast and prediction intervals at each time point.
        x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
        x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
        x_forecast_pi_higher = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

        # Calculate RMSE for the selected plotting variable
        rmse_forecast = root_mean_squared_error(x_test[:, idx_fp], x_forecast_median[:, idx_fp])
        rmse_forecasts.append(rmse_forecast)

        message = f"RMSE: {rmse_forecast:.2f} {units}\n"
        message += f"Initial:\n"
        message += f"* GLC: {initial_glucose:.0f} mM\n"
        message += f"* GLN: {initial_glutamine:.0f} mM\n"
        message += f"* VCD: {initial_vcd:.0f} cells/mL\n"

        # Plot the test values, forecast median, and prediction interval
        ax.set_title(f"Batch {batch_id_test}", fontsize=9)
        ax.scatter(t_test, x_test[:, idx_fp], label="Test", s=8, color="tab:orange", edgecolors="none", zorder=3)
        ax.plot(t_test, x_forecast_median[:, idx_fp], label=f"VAR({p})", color="black", zorder=4)
        ax.fill_between(t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_higher[:, idx_fp],
                        label=f"{prediction_interval:.0%} PI", color="tab:blue", edgecolors="none", alpha=0.3, zorder=2)

        # Display forecast error and initial conditions
        ax.text(x=0.02, y=0.96, s=message, va='top', ha='left', transform=ax.transAxes, fontname="monospace",
                fontsize=7)

        # Track the maximum value needed for consistent subplot limits
        y_max = max(y_max, np.max(x_test[:, idx_fp]), np.max(x_forecast_pi_higher[:, idx_fp]))


    # Apply consistent axis limits and formatting to all subplots
    y_range = y_max - y_min
    y_buffer = 0.05 * y_range

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_ylim(bottom=y_min - y_buffer, top=y_max + y_buffer)

            if i == axes.shape[0] - 1:
                axes[i, j].set_xlabel("Time [h]")
                axes[i, j].xaxis.set_major_locator(MultipleLocator(48))

            else:
                axes[i, j].set_xticks([])

            if j == 0:
                axes[i, j].set_ylabel(feature_plot)
            else:
                axes[i, j].set_yticks([])

            # Add the legend to only one subplot to avoid repetition
            if i == 0 and j == 0:
                axes[i, j].legend(loc="upper right", fontsize=7)


    fig.supxlabel("Time [h]")

    # Save the figure
    fig.savefig(os.path.join("figures", f"figure_01_p{p:02d}_N{n_simulations:02d}.png"))
    plt.close(fig)

    # Calculate and report the forecast error
    rmse_mean_var = np.mean(rmse_forecasts)
    rmse_sd_var = np.std(rmse_forecasts)

    record = (p, n_simulations, rmse_mean_var, rmse_sd_var)
    records.append(record)

cols_export = ["p", "N simulations", f"RMSE (mean) [{units}]", f"RMSE (SD) [{units}]"]
df_export = pd.DataFrame.from_records(records, columns=cols_export)
df_export.to_csv(os.path.join("tables", "table_02_VAR_summary.csv"), index=False)

#VAR non-linear

# Define training parameters
n_simulations = 500
#n_simulations = 5 for testing/ease of running
p_max = 5
cv_splits = 3
poly_degree_max = 3
knn_k_max = 5

# Fraction of simulated forecasts used to calculate prediction intervals
prediction_interval = 0.95
alpha = 1 - prediction_interval

# List of non-linear models to explore
model_types = ["poly", "knn"]

# Define our GridSearchCV parameters.
# We skip degree = 1 to save on compute time and to avoid repeating a purely linear model.
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(2, poly_degree_max + 1)])

# We skip k=1 and weights="distance" to save on compute time and to avoid overfitting.
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(2, knn_k_max + 1)],
                      kneighborsregressor__weights=["uniform"])

# Create a random number generator
rng = np.random.default_rng(0)

nrows = 5
ncols = 6

n_iters = len(model_types) * p_max * batch_ids.shape[0] * n_simulations
c_iter = 0
records = []

for model_type in model_types:
    # Evaluate NVAR models with different numbers of lagged values
    for p in range(1, p_max + 1):
        # Create a figure containing one subplot for each set of experimental initial conditions.
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12), dpi=300, layout="constrained", squeeze=False)

        y_min = 0
        y_max = np.max(df.loc[:, feature_plot])
        rmse_forecasts = []

        # Leave-one-batch-out cross-validation
        for batch_index, batch_id_test in enumerate(batch_ids):

            # Training batches contain all batches except the test batch
            batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])

            # Generate the lagged value dataset used to train NVAR(p)
            x_lag_train, x_cur_train, _, b_cur_train = generate_x_lag_cur_var(p=p, batch_ids=batch_ids_train, df=df,
                                                                              features=features)

            a_min = np.min(x_cur_train, axis=0)
            a_max = np.max(x_cur_train, axis=0)

            # Create the model
            if model_type == "poly":
                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    PolynomialFeatures(include_bias=False),
                    LinearRegression()
                )
                param_grid = param_grid_poly
            elif model_type == "knn":
                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    KNeighborsRegressor()
                )
                param_grid = param_grid_knn
            else:
                model = None
                param_grid = {}

            # Keep the batches separate in GridSearchCV
            gkf = GroupKFold(n_splits=cv_splits)
            model_gscv = GridSearchCV(model, param_grid=param_grid, cv=gkf, scoring="neg_root_mean_squared_error",
                                      return_train_score=True)

            # Fit the model
            model_gscv.fit(x_lag_train, x_cur_train, groups=b_cur_train)


            # Calculate residual errors on the training data.
            # These residuals will later be sampled during forecast simulations to estimate white noise.
            x_cur_train_pred = model_gscv.predict(x_lag_train)
            x_cur_train_res = x_cur_train_pred - x_cur_train

            # Initial state containing all process variables
            idx_ic = df_ic.index[df_ic["Batch ID"] == batch_id_test][0]
            initial_glucose = df_ic.at[idx_ic, "Glucose [mM]"]
            initial_glutamine = df_ic.at[idx_ic, "Glutamine [mM]"]
            initial_vcd = df_ic.at[idx_ic, "VCD [cells/mL]"]

            x_initial = np.array([initial_glucose, initial_glutamine, 0, 0, initial_vcd, initial_vcd, 0])

            # Determine which subplot corresponds to the test batch initial conditions
            i_ax = batch_index // ncols
            j_ax = batch_index % ncols
            ax = axes[i_ax, j_ax]

            # Extract the data for the test batch
            mask = df.loc[:, "Batch ID"] == batch_id_test
            t_test = df.loc[mask, "Time [h]"].to_numpy()
            x_test = df.loc[mask, features].to_numpy()

            # Perform multiple forecast simulations to estimate uncertainty
            x_forecasts = []
            for _ in range(n_simulations):
                # Initialize the forecast using the known initial conditions.
                x_forecast = [x_initial]

                # Generate forecasts recursively.
                while len(x_forecast) < t_test.shape[0]:

                    x_lag = x_forecast[::-1][:p].copy()

                    # At the beginning of the simulation there may be fewer than p available lagged vectors.
                    # If necessary, repeat the most recent state until p lagged vectors are available.
                    while len(x_lag) < p:
                        x_lag.append(x_lag[-1])

                    # Concatenate all lagged vectors into a single feature vector expected by the NVAR model.
                    x_lag = np.concatenate(x_lag, axis=0)
                    x_lag = np.expand_dims(x_lag, axis=0)

                    # Sample residual errors from the training data.
                    # This assumes the residuals behave like white noise.
                    white_noise = []
                    for j in range(x_cur_train_res.shape[1]):
                        white_noise.append(rng.choice(x_cur_train_res[:, j]))
                    white_noise = np.array([white_noise])

                    # Predict the next process state baseline
                    x_cur_baseline = model_gscv.predict(x_lag)

                    # Don't let the model predict values outside the range it has been trained on.
                    # k-NN has this behaviour by default. Polynomials do not.
                    x_cur_baseline = np.clip(x_cur_baseline, a_min=a_min, a_max=a_max)

                    # Add white noise
                    x_cur = x_cur_baseline + white_noise

                    # Prevent negative concentration predictions after white noise is added
                    x_cur = np.clip(x_cur[0], a_min=0, a_max=None)

                    x_forecast.append(x_cur)

                x_forecast = np.array(x_forecast)

                x_forecasts.append(x_forecast)

                c_iter += 1
                progress_bar(c_iter, n_iters)

            x_forecasts = np.array(x_forecasts)

            # Calculate the median forecast and prediction intervals at each time point.
            x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
            x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
            x_forecast_pi_higher = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

            # Calculate RMSE for the selected plotting variable
            rmse_forecast = root_mean_squared_error(x_test[:, idx_fp], x_forecast_median[:, idx_fp])
            rmse_forecasts.append(rmse_forecast)

            message = f"RMSE: {rmse_forecast:.2f} {units}\n"
            message += f"Initial:\n"
            message += f"* GLC: {initial_glucose:.0f} mM\n"
            message += f"* GLN: {initial_glutamine:.0f} mM\n"
            message += f"* VCD: {initial_vcd:.0f} cells/mL\n"

            # Extract the best hyperparameters
            if model_type == "poly":
                best_degree = model_gscv.best_params_["polynomialfeatures__degree"]
                message += f"Degree: {best_degree}"
            elif model_type == "knn":
                best_n_neighbors = model_gscv.best_params_["kneighborsregressor__n_neighbors"]
                best_weight = model_gscv.best_params_["kneighborsregressor__weights"]
                message += f"k={best_n_neighbors}, {best_weight}"

            # Plot the test values, forecast median, and prediction interval
            ax.scatter(t_test, x_test[:, idx_fp], label="Test", s=10, color="tab:orange", edgecolors="black", zorder=3)
            ax.plot(t_test, x_forecast_median[:, idx_fp], label=f"NVAR({p})", color="black", zorder=4)
            ax.fill_between(t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_higher[:, idx_fp],
                            label=f"{prediction_interval:.0%} PI", color="tab:blue", edgecolors="none", alpha=0.3,
                            zorder=2)

            # Display forecast error and initial conditions
            ax.text(x=0.02, y=0.96, s=message, va='top', ha='left', transform=ax.transAxes, fontname="monospace",
                    fontsize=7)

            # Track the maximum value needed for consistent subplot limits
            y_max = max(y_max, np.max(x_test[:, idx_fp]), np.max(x_forecast_pi_higher[:, idx_fp]))

        # Apply consistent axis limits and formatting to all subplots
        y_range = y_max - y_min
        y_buffer = 0.05 * y_range

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i, j].set_ylim(bottom=y_min - y_buffer, top=y_max + y_buffer)

                if i == axes.shape[0] - 1:
                    axes[i, j].set_xlabel("Time [h]")
                    axes[i, j].xaxis.set_major_locator(MultipleLocator(48))

                else:
                    axes[i, j].set_xticks([])

                if j == 0:
                    axes[i, j].set_ylabel(feature_plot)
                else:
                    axes[i, j].set_yticks([])

                # Add the legend to only one subplot to avoid repetition
                if i == 0 and j == 0:
                    axes[i, j].legend(loc="upper right", fontsize=7)

        # Save the figure
        fig.savefig(os.path.join("figures", f"figure_02_{model_type}_p{p:02d}_N{n_simulations:04d}.png"))
        plt.close(fig)

        # Calculate and report the forecast error
        rmse_mean = np.mean(rmse_forecasts)
        rmse_sd = np.std(rmse_forecasts)

        record = (model_type, p, n_simulations, rmse_mean, rmse_sd)
        records.append(record)

cols_export = ["Model", "p", "N simulations", f"RMSE (mean) [{units}]", f"RMSE (SD) [{units}]"]
df_export = pd.DataFrame.from_records(records, columns=cols_export)
df_export.to_csv(os.path.join("tables", "table_03_NVAR_summary.csv"), index=False)

# ============================================================================
# WHITE NOISE ANALYSIS (NVAR)
# ============================================================================

# Define parameters
prediction_interval = 0.95
alpha = 1 - prediction_interval

model_types = ["poly", "knn"]

# Evaluate residuals for each nonlinear model
for model_type in model_types:

    for p in range(1, p_max + 1):

        # Store residuals and corresponding times
        residuals_all = []
        times_all = []

        # Leave-one-batch-out cross-validation
        for batch_id_test in batch_ids:

            batch_ids_train = np.array([
                i for i in batch_ids if i != batch_id_test
            ])

            # Generate training data
            x_lag_train, x_cur_train, _, b_cur_train = generate_x_lag_cur_var(
                p=p,
                batch_ids=batch_ids_train,
                df=df,
                features=features
            )

            # Create model
            if model_type == "poly":

                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    PolynomialFeatures(include_bias=False),
                    LinearRegression()
                )

                param_grid = param_grid_poly

            elif model_type == "knn":

                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    KNeighborsRegressor()
                )

                param_grid = param_grid_knn

            # Cross-validation
            gkf = GroupKFold(n_splits=cv_splits)

            model_gscv = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=gkf,
                scoring="neg_root_mean_squared_error",
                return_train_score=True
            )

            # Fit model
            model_gscv.fit(
                x_lag_train,
                x_cur_train,
                groups=b_cur_train
            )

            # Calculate residuals
            x_cur_train_pred = model_gscv.predict(x_lag_train)

            x_cur_train_res = x_cur_train_pred - x_cur_train

            # Extract corresponding time points
            _, _, t_cur_train, _ = generate_x_lag_cur_var(
                p=p,
                batch_ids=batch_ids_train,
                df=df,
                features=features
            )

            residuals_all.append(x_cur_train_res)
            times_all.append(t_cur_train)

        # Combine residuals from all training batches
        x_cur_res = np.concatenate(residuals_all, axis=0)
        t_cur = np.concatenate(times_all, axis=0)

        # Determine unique time points
        t_unique = np.unique(t_cur)

        # Calculate residual statistics for mAb
        res_mean = []
        res_pi_upper = []
        res_pi_lower = []

        for t in t_unique:

            mask = t_cur == t
            x_cur_res_t = x_cur_res[mask, idx_fp]

            n_mean = np.mean(x_cur_res_t)
            n_sd = np.std(x_cur_res_t, ddof=1)

            dof = len(x_cur_res_t) - 1

            if dof > 0:

                q = alpha / 2
                t_critical = stats.t.isf(q=q, df=dof)
                # Approximate prediction interval for an individual residual,
                # not a confidence interval for the mean residual.
                prediction_half_width = t_critical * n_sd * np.sqrt(1 + 1 / len(x_cur_res_t))

                res_mean.append(n_mean)
                res_pi_upper.append(n_mean + prediction_half_width)
                res_pi_lower.append(n_mean - prediction_half_width)

            else:
                res_mean.append(n_mean)
                res_pi_upper.append(n_mean)
                res_pi_lower.append(n_mean)

        res_mean = np.array(res_mean)
        res_pi_upper = np.array(res_pi_upper)
        res_pi_lower = np.array(res_pi_lower)

        # Determine plotting limits
        y_max = max(
            np.max(np.abs(res_mean)),
            np.max(np.abs(res_pi_upper)),
            np.max(np.abs(res_pi_lower)),
            np.max(np.abs(x_cur_res[:, idx_fp]))
        )

        fig, ax = plt.subplots(
            1, 1,
            figsize=(14.39 / 2.54, 12.09 / 2.54),
            dpi=300,
            layout="constrained"
        )

        # Plot residuals
        ax.scatter(
            t_cur,
            x_cur_res[:, idx_fp],
            label="Noise",
            color="tab:orange",
            s=5,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.7,
            zorder=3
        )

        ax.plot(
            t_unique,
            res_mean,
            label="Mean",
            color="black",
            zorder=4
        )

        ax.fill_between(
            t_unique,
            res_pi_lower,
            res_pi_upper,
            label=f"{prediction_interval:.0%} PI",
            color="tab:blue",
            edgecolors="none",
            alpha=0.3,
            zorder=2
        )

        ax.axhline(
            y=0,
            color="black",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5
        )

        ax.set_xlabel("Time [h]", fontsize=14)
        ax.set_ylabel(f"{feature_plot} (Residuals)", fontsize=14)

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(loc="upper left", fontsize=12)

        ax.xaxis.set_major_locator(MultipleLocator(48))

        ax.set_xlim(left=float(np.min(t_cur)), right=float(np.max(t_cur)))
        ax.set_ylim(
            bottom=-1.05 * y_max,
            top=1.05 * y_max
        )

        # Save figure
        fig.savefig(
            os.path.join(
                "figures",
                f"figure_07_{model_type}_p{p:02d}_white_noise.png"
            )
        )

        plt.close(fig)

        print(f"White noise analysis completed: {model_type}, p={p}")



# =======================
#    Forecast function
# =======================

def simulate_trajectories(initial_glucose, initial_glutamine, initial_vcd,
                          model, residuals, p, features, n_simulations,
                          rng, n_steps, a_min, a_max):

    expected_features = [
        "Glucose [mM]", "Glutamine [mM]", "Lactate [mM]",
        "Ammonia [mM]", "VCD [cells/mL]", "TCD [cells/mL]", "mAb [mg/L]"
    ]
    if list(features) != expected_features:
        raise ValueError(
            "The feature columns/order do not match the initial-state vector. "
            f"Expected {expected_features}, received {list(features)}."
        )

    x_initial = np.array([
        initial_glucose, initial_glutamine, 0.0, 0.0,
        initial_vcd, initial_vcd, 0.0
    ], dtype=float)

    forecasts = np.zeros((n_simulations, n_steps, len(features)), dtype=float)

    for simulation_id in range(n_simulations):
        # Repeat the known initial state to provide p lagged observations.
        history = [x_initial.copy() for _ in range(p)]
        for step in range(n_steps):
            x_lag = np.concatenate(history[-p:][::-1]).reshape(1, -1)
            baseline = model.predict(x_lag)[0]

            # Keep baseline predictions within the training data range
            baseline = np.clip(baseline, a_min=a_min, a_max=a_max)

            # Sample one residual independently for each process variable
            noise_indices = rng.integers(
                0, residuals.shape[0],
                size=residuals.shape[1]
            )

            white_noise = residuals[
                noise_indices, np.arange(residuals.shape[1])
            ]

            # Add noise and constrain predictions to the training data range
            next_state = np.clip(
                baseline + white_noise,
                a_min=np.maximum(a_min, 0),
                a_max=a_max
            )

            forecasts[simulation_id, step, :] = next_state
            history.append(next_state)

    return forecasts


# ======================
#   Model optimization
# ======================

# Select the final model
model_type = "poly"

# Select the optimal number of lags
p = 3

# For polynomial NVAR, set the selected polynomial degree
best_degree = 2

# For kNN NVAR, set the selected number of neighbours
best_k = 3

# Generate lagged data using all 30 batches
x_lag_all, x_cur_all, t_cur_all, b_cur_all = generate_x_lag_cur_var(
    p=p,
    batch_ids=batch_ids,
    df=df,
    features=features
)
# Determine the minimum and maximum training values for each variable
a_min = np.min(x_cur_all, axis=0)
a_max = np.max(x_cur_all, axis=0)
# Create the selected model
if model_type == "var":

    model = make_pipeline(
        StandardScalerVAR(p=p),
        LinearRegression()
    )

elif model_type == "poly":

    model = make_pipeline(
        StandardScalerVAR(p=p),
        PolynomialFeatures(degree=best_degree, include_bias=False),
        LinearRegression()
    )

elif model_type == "knn":

    model = make_pipeline(
        StandardScalerVAR(p=p),
        KNeighborsRegressor(n_neighbors=best_k, weights="uniform")
    )

else:
    raise ValueError("model_type must be 'var', 'poly', or 'knn'.")

# Fit the final model
model.fit(x_lag_all, x_cur_all)

# Calculate residuals
x_cur_all_pred = model.predict(x_lag_all)
x_cur_all_res = x_cur_all - x_cur_all_pred

# ==================
# Optimization
# ==================

# Initial condition ranges
glucose_levels = np.arange(5, 36, 5)
glutamine_levels = np.arange(1, 8, 1)
vcd_levels = np.arange(200000, 1400001, 200000)

# Forecasting parameters
n_simulations_opt = 500
prediction_interval = 0.95
alpha = 1 - prediction_interval

# Harvest times
# Measurements are taken every 8 hours, up to 672 hours
n_steps = 84

times_h = np.arange(8, 673, 8)
times_d = times_h / 24

# Downtime between batches
downtime = 1

# Identify the mAb column
idx_mab = list(features).index("mAb [mg/L]")

# Random number generator
rng = np.random.default_rng(0)

# ============================================
# Economic Optimization
# ============================================

optimization_results = []

for initial_glucose in glucose_levels:

    for initial_glutamine in glutamine_levels:

        for initial_vcd in vcd_levels:

            # Calculate cell culture cost
            cell_culture_cost = (
                10 * initial_glucose
                + 20 * initial_glutamine
                + 0.0005 * initial_vcd
            )

            # Generate forecast simulations
            x_forecasts = simulate_trajectories(
                initial_glucose=initial_glucose,
                initial_glutamine=initial_glutamine,
                initial_vcd=initial_vcd,
                model=model,
                residuals=x_cur_all_res,
                p=p,
                features=features,
                n_simulations=500,
                rng=rng,
                n_steps=n_steps,
                a_min=a_min,
                a_max=a_max
            )

            # Extract mAb predictions
            mab_forecasts = x_forecasts[:, :, idx_mab]

            # Calculate cell culture value
            cell_culture_value = 1000 * mab_forecasts

            # Calculate profit
            profit = cell_culture_value - cell_culture_cost

            # Calculate space-time profit
            space_time_profit = profit / (times_d + downtime)

            # Calculate space-time yield
            space_time_yield = mab_forecasts / (times_d + downtime)

            # Calculate median performance across simulations
            median_profit = np.median(space_time_profit, axis=0)
            median_yield = np.median(space_time_yield, axis=0)

            # Identify optimal harvest times
            idx_best_profit = np.argmax(median_profit)
            idx_best_yield = np.argmax(median_yield)

            # Store the optimal results
            optimization_results.append({

                "Glucose [mM]": initial_glucose,
                "Glutamine [mM]": initial_glutamine,
                "VCD [cells/mL]": initial_vcd,

                "Cell Culture Cost [$/L]": cell_culture_cost,

                "Profit Harvest Time [d]": times_d[idx_best_profit],
                "Maximum Space-Time Profit [$/L/d]": median_profit[idx_best_profit],

                "Yield Harvest Time [d]": times_d[idx_best_yield],
                "Maximum Space-Time Yield [mg/L/d]": median_yield[idx_best_yield]

            })

# Convert results to a DataFrame
df_optimization = pd.DataFrame(optimization_results)

# Sort by space-time profit
df_optimization = df_optimization.sort_values(
    "Maximum Space-Time Profit [$/L/d]",
    ascending=False
)

# Export results
os.makedirs("tables", exist_ok=True)

df_optimization.to_csv(
    os.path.join("tables", "table_04_optimization_results.csv"),
    index=False
)

print("\nOptimization completed.")

# ===========================================
# Calculating optimal intial conditions
# ===========================================

def summarize_optimum(row, objective):

    # Extract optimal initial conditions
    initial_glucose = row["Glucose [mM]"]
    initial_glutamine = row["Glutamine [mM]"]
    initial_vcd = row["VCD [cells/mL]"]

    # Determine optimal harvest time
    if objective == "profit":
        harvest_time = row["Profit Harvest Time [d]"]

    else:
        harvest_time = row["Yield Harvest Time [d]"]

    # Calculate cell culture cost
    cell_culture_cost = (
        10 * initial_glucose
        + 20 * initial_glutamine
        + 0.0005 * initial_vcd
    )

    # Run additional simulations at the selected conditions
    x_forecasts = simulate_trajectories(
        initial_glucose=initial_glucose,
        initial_glutamine=initial_glutamine,
        initial_vcd=initial_vcd,
        model=model,
        residuals=x_cur_all_res,
        p=p,
        features=features,
        n_simulations=500,
        rng=rng,
        n_steps=n_steps,
        a_min=a_min,
        a_max=a_max
    )

    # Find the harvest time index
    idx_harvest = np.argmin(np.abs(times_d - harvest_time))

    # Extract mAb predictions at harvest
    mab_at_harvest = x_forecasts[:, idx_harvest, idx_mab]

    # Calculate cell culture value
    culture_value = 1000 * mab_at_harvest

    # Calculate space-time profit and yield
    st_profit = (
        culture_value - cell_culture_cost
    ) / (harvest_time + downtime)

    st_yield = mab_at_harvest / (harvest_time + downtime)

    # Calculate 95% prediction intervals
    lower = alpha / 2
    upper = 1 - alpha / 2

    results = {

        "Objective": objective,

        "Glucose [mM]": initial_glucose,
        "Glutamine [mM]": initial_glutamine,
        "VCD [cells/mL]": initial_vcd,

        "Harvest Time [d]": harvest_time,

        "Cell Culture Cost [$/L]": cell_culture_cost,

        "Cell Culture Value Median [$/L]":
            np.median(culture_value),

        "Cell Culture Value Lower 95% [$/L]":
            np.quantile(culture_value, lower),

        "Cell Culture Value Upper 95% [$/L]":
            np.quantile(culture_value, upper),

        "Space-Time Profit Median [$/L/d]":
            np.median(st_profit),

        "Space-Time Profit Lower 95% [$/L/d]":
            np.quantile(st_profit, lower),

        "Space-Time Profit Upper 95% [$/L/d]":
            np.quantile(st_profit, upper),

        "Space-Time Yield Median [mg/L/d]":
            np.median(st_yield),

        "Space-Time Yield Lower 95% [mg/L/d]":
            np.quantile(st_yield, lower),

        "Space-Time Yield Upper 95% [mg/L/d]":
            np.quantile(st_yield, upper)

    }

    return results

# Identify the optimal conditions for profit
best_profit_row = df_optimization.loc[
    df_optimization["Maximum Space-Time Profit [$/L/d]"].idxmax()
]

# Identify the optimal conditions for yield
best_yield_row = df_optimization.loc[
    df_optimization["Maximum Space-Time Yield [mg/L/d]"].idxmax()
]

profit_results = summarize_optimum(best_profit_row, "profit")
yield_results = summarize_optimum(best_yield_row, "yield")

# Combine results
df_optimal = pd.DataFrame([
    profit_results,
    yield_results
])

df_optimal.to_csv(
    os.path.join("tables", "table_05_optimal_conditions.csv"),
    index=False
)

print("\nOPTIMAL SPACE-TIME PROFIT")
print(pd.Series(profit_results))

print("\nOPTIMAL SPACE-TIME YIELD")
print(pd.Series(yield_results))


# #economics
# cell_culture_cost=10*(x_initial[0])+20*(x_initial[1])+0.0005*(x_initial[2])
# # predicted value from output model: cell_culture_value=1000*(mAb)
# profit=(cell_culture_value-cell_culture_cost)
# space_time_profit=(profit / harvest_time)+down_time
# space_time_yield=(mAb / harvest_time) + down_time


