# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C")

#find initial conditions and output them from the constants tab and print them at the beginning
#three things to test: non-linear VAR (polynomial and knn), linear VAR, AR (if I feel like it/if I have time)

import numpy as np


class StandardScalerAR:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None

        self.n_samples_seen_ = None
        self.n_features_in_ = None

        self.feature_names_in_ = None

    def fit(self, x, y=None):
        self.mean_ = np.mean(x)
        self.scale_ = np.std(x)
        self.var_ = np.var(x)

        self.n_samples_seen_ = x.shape[0]
        self.n_features_in_ = x.shape[1]

        return self

    def transform(self, x):
        x_tr = (x - self.mean_) / self.scale_
        return x_tr

    def inverse_transform(self, x):
        x_original = self.scale_ * x + self.mean_
        return x_original

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class StandardScalerVAR:

    def __init__(self, p):
        self.p = p

        self.mean_ = None
        self.scale_ = None
        self.var_ = None

        self.n_samples_seen_ = None
        self.n_features_in_ = None

        self.feature_names_in_ = None

    def fit(self, x, y=None):
        n_var_features = x.shape[1] // self.p

        assert n_var_features * self.p == x.shape[1]
        x_sf = []
        for i in range(self.p):
            idx_start = i * n_var_features
            idx_end = idx_start + n_var_features
            x_sf.append(x[:, idx_start:idx_end])

        x_sf = np.concatenate(x_sf, axis=0)

        x_sf_mean = np.mean(x_sf, axis=0)
        x_sf_sd = np.std(x_sf, axis=0)
        x_sf_var = np.var(x_sf, axis=0)

        self.mean_ = np.tile(x_sf_mean, (1, self.p))
        self.scale_ = np.tile(x_sf_sd, (1, self.p))
        self.var_ = np.tile(x_sf_var, (1, self.p))

        self.n_samples_seen_ = x.shape[0]
        self.n_features_in_ = x.shape[1]

        return self

    def transform(self, x):
        x_tr = (x - self.mean_) / self.scale_
        return x_tr

    def inverse_transform(self, x):
        x_original = self.scale_ * x + self.mean_
        return x_original

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def get_params(self, deep=True):
        return {"p": self.p}

    def set_params(self, **params):
        if "p" in params:
            self.p = params["p"]

        return self