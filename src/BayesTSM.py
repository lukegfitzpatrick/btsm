import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import theano
import theano.tensor as tt
from fbprophet import Prophet

class btsm:

    def __init__(self,\
                 data,\
                 n_changepoints=25,\
                 growth_prior_scale=0.05,\
                 jump_prior_scale=0.05,\
                 changepoint_range=0.8):

        self.df = self.prep_data(data)
        self.n_changepoints = n_changepoints
        self.growth_prior_scale=growth_prior_scale
        self.jump_prior_scale = jump_prior_scale
        self.changepoint_range = changepoint_range




    def seasonality_model(self, m, df, period='yearly', seasonality_prior_scale=25):

        if period == 'yearly':
            n = 10
            # rescale the period, as t is also scaled
            p = 365.25 / (df['ds'].max() - df['ds'].min()).days
        else:  # weekly
            n = 3
            # rescale the period, as t is also scaled
            p = 7 / (df['ds'].max() - df['ds'].min()).days
        x = self.fourier_series(df['t'], p, n)
        with m:
            beta = pm.Normal(f'beta_{period}', mu=0, sd=seasonality_prior_scale, shape=2 * n)
        return x, beta

    def fourier_series(self, t, p=365.25, n=10):
    # 2 pi n / p
        x = 2 * np.pi * np.arange(1, n + 1) / p
        # 2 pi n / p * t
        x = x * t[:, None]
        x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        return x

    def fit_seasonal_model(self):
        m = pm.Model()

        with m:
            # changepoints_prior_scale is None, so the exponential distribution
            # will be used as prior on \tau.
            #y, A, s = trend_model(m, df['t'], changepoints_prior_scale=None)
            x_yearly, beta_yearly = self.seasonality_model(m, self.df, 'yearly')
            x_weekly, beta_weekly = self.seasonality_model(m, self.df, 'weekly')

            seasonal = self.det_dot(x_yearly, beta_yearly) + self.det_dot(x_weekly, beta_weekly)

            sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
            obs = pm.Normal('obs',
                         mu=seasonal,
                         sd=sigma,
                         observed=self.df['y_scaled'])
        with m:
            aprox = pm.find_MAP()

        self.wkly_ses = self.det_dot(x_weekly, aprox['beta_weekly'])
        self.yrly_ses = self.det_dot(x_yearly, aprox['beta_yearly'])
        return (self.det_dot(x_yearly, aprox['beta_yearly'].T) + self.det_dot(x_weekly, aprox['beta_weekly'].T))*self.df['y'].max()




    def fit_linear_model(self):
            m = pm.Model()

            with m:
                y, A, s = self.trend_model(m,\
                                           self.df['t'],\
                                           n_changepoints = self.n_changepoints,\
                                           changepoint_range = self.changepoint_range,\
                                           jump_prior_scale = self.jump_prior_scale,\
                                           growth_prior_scale = self.growth_prior_scale)

                sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
                pm.Normal('obs',
                             mu=y,
                             sd=sigma,
                             observed=self.df['y_scaled'])

            with m:
                aprox = pm.find_MAP()
                self.trace = pm.sample(200)
                self.aprox = aprox
            g1= self.det_trend(aprox['k'], aprox['m'], aprox['delta'], self.df['t'], s, A) * self.df['y'].max()

            return g1


    def prep_data(self, df):
        df['observed']  = df['y']
        df['y_scaled'] = df['y']/df['y'].max()
        df['ds'] = pd.to_datetime(df['Date'])
        df['t'] = (df['ds'] - df['ds'].min()) / (df['ds'].max() - df['ds'].min())

        return df



    def det_dot(self, a, b):
        return (a * b[None, :]).sum(axis=-1)
    """
    The theano dot product and NUTS sampler don't work with large matrices?

    :param a: (np matrix)
    :param b: (theano vector)
    """


    def det_trend(self, k, m, delta, t, s, A):
        return (k + np.dot(A, delta)) * t + (m + np.dot(A, (-s * delta)))

    def trend_model(self, m, t, n_changepoints, jump_prior_scale,
                growth_prior_scale, changepoint_range):
        s = np.linspace(0, changepoint_range * np.max(t), n_changepoints + 1)[1:]

        # * 1 casts the boolean to integers
        A = (t[:, None] > s) * 1

        with m:
            # initial growth
            k = pm.Normal('k', 0 , growth_prior_scale)

            if jump_prior_scale is None:
                jump_prior_scale = pm.Exponential('tau', 1.5)

            # rate of change
            delta = pm.Laplace('delta', 0, jump_prior_scale, shape=n_changepoints)
            # offset
            m = pm.Normal('m', 0, 0.25)
            gamma = -s * delta

            g = (k + self.det_dot(A, delta)) * t + (m + self.det_dot(A, gamma))
        return g, A, s
