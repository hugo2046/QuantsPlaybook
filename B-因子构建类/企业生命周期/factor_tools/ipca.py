'''
Author: shen.lan123@gmail.com
Date: 2022-04-27 23:07:26
LastEditTime: 2022-04-28 14:19:10
LastEditors: Please set LastEditors
Description: 
This is a Python implementation of the Instrumtented Principal Components Analysis framework by Kelly, Pruitt, Su (2017).
https://github.com/bkelly-lab/ipca
'''
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from numba import jit
import pandas as pd
import numpy as np
import scipy as sp
import progressbar
import warnings
import time


class InstrumentedPCA(BaseEstimator):
    """
    This class implements the IPCA algorithm by Kelly, Pruitt, Su (2017).
    Parameters
    ----------
    n_factors : int, default=1
        The total number of factors to estimate. Note, the number of
        estimated factors is automatically reduced by the number of
        pre-specified factors. For example, if n_factors = 2 and one
        pre-specified factor is passed, then InstrumentedPCA will estimate
        one factor estimated in addition to the pre-specified factor.
    intercept : boolean, default=False
        Determines whether the model is estimated with or without an intercept
    max_iter : int, default=10000
        Maximum number of alternating least squares updates before the
        estimation is stopped
    iter_tol : float, default=10e-6
        Tolerance threshold for stopping the alternating least squares
        procedure
    alpha : scalar
        Regularizing constant for Gamma estimation.  If this is set to
        zero then the estimation defaults to non-regularized.
    l1_ratio : scalar
        Ratio of l1 and l2 penalties for elastic net Gamma fit.
    n_jobs : scalar
        number of jobs for F step estimation in ALS, if set to one no
        parallelization is done
    backend : str
        label for Joblib backend used for F step in ALS
    """
    def __init__(self,
                 n_factors=1,
                 intercept=False,
                 max_iter=10000,
                 iter_tol=10e-6,
                 alpha=0.,
                 l1_ratio=1.,
                 n_jobs=1,
                 backend="loky"):

        # paranoid parameter checking to make it easier for users to know when
        # they have gone awry and to make it safe to assume some variables can
        # only have certain settings
        if not isinstance(n_factors, int) or n_factors < 1:
            raise ValueError('n_factors must be an int greater / equal 1.')
        if not isinstance(intercept, bool):
            raise NotImplementedError('intercept must be  boolean')
        if not isinstance(iter_tol, float) or iter_tol >= 1:
            raise ValueError('Iteration tolerance must be smaller than 1.')
        if l1_ratio > 1. or l1_ratio < 0.:
            raise ValueError("l1_ratio must be between 0 and 1")
        if alpha < 0.:
            raise ValueError("alpha must be greater than or equal to 0")

        # Save parameters to the object
        params = locals()
        for k, v in params.items():
            if k != 'self':
                setattr(self, k, v)

    def fit(self,
            X,
            y,
            indices=None,
            PSF=None,
            Gamma=None,
            Factors=None,
            data_type="portfolio",
            label_ind=False,
            **kwargs):
        """
        Fits the regressor to the data using an alternating least squares
        scheme.
        Parameters
        ----------
        X :  numpy array or pandas DataFrame
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.
            If given as a DataFrame, we assume that it contains a MutliIndex
            mapping to each entity-time pair
        y : numpy array or pandas Series
            dependent variable where indices correspond to those in X
            If given as a Series, we assume that it contains a MutliIndex
            mapping to each entity-time pair
        indices : numpy array, optional
            array containing the panel indices.  Should consist of two
            columns:
            - Column 1: entity id (i)
            - Column 2: time index (t)
            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
        PSF : numpy array, optional
            Set of pre-specified factors as matrix of dimension (M, T)
        Gamma : numpy array, optional
            If provided, starting values for Gamma (see Notes)
        Factors : numpy array
            If provided, starting values for Factors (see Notes)
        data_type : str
            label for data-type used for ALS estimation, one of the following:
            1. panel
            ALS uses the untransformed X and y for the estimation.
            This is currently marginally slower than the portfolio estimation
            but is necessary when performing regularized estimation
            (alpha > 0).
            2. portfolio
            ALS uses a matrix of characteristic weighted portfolios (Q)
            as well as a matrix of weights (W) and count of non-missing
            observations for each time period (val_obs) for the estimation.
            See _build_portfolio for details on how these variables are formed
            from the initial X and y.
            Currently, the bootstrap procedure is only implemented in terms
            of the portfolio data_type.
        Returns
        -------
        self
        Notes
        -----
        Updates InstrumentedPCA instances to include param estimates:
        Gamma : numpy array
            Array with dimensions (L, n_factors) containing the
            mapping between characteristics and factors loadings. If there
            are M many pre-specified factors in the model then the
            matrix returned is of dimension (L, (n_factors+M)).
            If an intercept is included in the model, its loadings are
            returned in the last column of Gamma.
        Factors : numpy array
            Array with dimensions (n_factors, T) containing the estimated
            factors. If pre-specified factors were passed the returned
            array is of dimension ((n_factors - M), T),
            corresponding to the n_factors - M many factors estimated on
            top of the pre-specified ones.
        """

        # handle input
        X, y, indices, metad = _prep_input(X, y, indices)
        N, L, T = metad["N"], metad["L"], metad["T"]

        # set data_type to panel if doing regularized estimation
        if self.alpha > 0.:
            data_type = "panel"

        # Handle pre-specified factors
        if PSF is not None:
            if np.size(PSF, axis=1) != np.size(metad["dates"]):
                raise ValueError("""Number of PSF observations must match
                                 number of unique dates""")
            self.has_PSF = True
        else:
            self.has_PSF = False

        if self.has_PSF:
            if np.size(PSF, axis=0) == self.n_factors:
                print("""Note: The number of factors (n_factors) to be
                      estimated matches the number of
                      pre-specified factors. No additional factors
                      will be estimated. To estimate additional
                      factors increase n_factors.""")

        #  Treating intercept as if was a prespecified factor
        if self.intercept:
            self.n_factors_eff = self.n_factors + 1
            if PSF is not None:
                PSF = np.concatenate((PSF, np.ones((1, T))), axis=0)
            elif PSF is None:
                PSF = np.ones((1, T))
        else:
            self.n_factors_eff = self.n_factors

        # Check that enough features provided
        if np.size(X, axis=1) < self.n_factors_eff:
            raise ValueError("""The number of factors requested exceeds number
                              of features""")

        # Determine fit case - if intercept or PSF or both use PSFcase fitting
        # Note PSFcase in contrast to has_PSF is only indicating
        # that the IPCA fitting is carried out as if PSF were passed even if
        # only an intercept was passed.
        self.PSFcase = True if self.has_PSF or self.intercept else False

        # store data
        self.X, self.y, self.indices, self.PSF = X, y, indices, PSF
        Q, W, val_obs = _build_portfolio(X, y, indices, metad)
        self.Q, self.W, self.val_obs = Q, W, val_obs
        self.metad = metad

        # Run IPCA
        Gamma, Factors = self._fit_ipca(X=X,
                                        y=y,
                                        indices=indices,
                                        Q=Q,
                                        W=W,
                                        val_obs=val_obs,
                                        PSF=PSF,
                                        Gamma=Gamma,
                                        Factors=Factors,
                                        data_type=data_type,
                                        **kwargs)

        # Store estimates
        if self.PSFcase:
            date_ln = len(metad["dates"])
            if self.intercept and self.has_PSF:
                PSF = np.concatenate((PSF, np.ones((1, date_ln))), axis=0)
            elif self.intercept:
                PSF = np.ones((1, date_ln))
            if Factors is not None:
                Factors = np.concatenate((Factors, PSF), axis=0)
            else:
                Factors = PSF

        self.Gamma, self.Factors = Gamma, Factors

        return self

    def get_factors(self, label_ind=False):
        """returns a tuple containing Gamma and Factors
        Parameters
        ----------
        label_ind : bool
            if provided we return the factors as pandas DataFrames with
            index info applied
        Returns
        -------
        tuple
            containing Gamma and Factors
        """

        Gamma, Factors = self.Gamma.copy(), self.Factors.copy()
        if label_ind:
            Gamma = pd.DataFrame(Gamma, index=self.metad["chars"])
            Factors = pd.DataFrame(Factors, columns=self.metad["dates"])

        return Gamma, Factors

    def fit_path(self,
                 X,
                 y,
                 indices=None,
                 PSF=None,
                 alpha_l=None,
                 n_splits=10,
                 split_method=GroupKFold,
                 n_jobs=1,
                 backend="loky",
                 **kwargs):
        """Fit a path of elastic net fits for various regularizing constants
        Parameters
        ----------
        X :  numpy array or pandas DataFrame
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.
            If given as a DataFrame, we assume that it contains a mutliindex
            mapping to each entity-time pair
        y : numpy array or pandas Series
            dependent variable where indices correspond to those in X
            If given as a Series, we assume that it contains a mutliindex
            mapping to each entity-time pair
        indices : numpy array, optional
            array containing the panel indices.  Should consist of two
            columns:
            - Column 1: entity id (i)
            - Column 2: time index (t)
            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
        PSF : numpy array, optional
            Set of pre-specified factors as matrix of dimension (M, T)
        alpha_l : iterable, optional
            list of regularizing constants to use for path
        n_splits : scalar
            number of CV partitions
        split_method : sklearn cross-validation generator factory
            method to generate CV partitions
        n_jobs : scalar
            number of jobs for parallel CV estimation
        backend : str
            label for joblib backend
        Returns
        -------
        cvmse : numpy matrix
            array of dim (P x (C + 1)) where P is the number of reg
            constants and C is the number of CV partitions
        """

        # handle input
        X, y, indices, metad = _prep_input(X, y, indices)

        # handle data type, since we are doing regularized estimation
        # only the panel fit makes sense here
        if "data_type" in kwargs:
            data_type = kwargs.pop("data_type")
        else:
            data_type = "panel"
        if data_type == "portfolio":
            raise ValueError("Unsupported data_type for fit_path: \
                              'portfolio'. Regularized estimation is only \
                              implemented for 'panel' data_type currently")

        # init alphas
        if alpha_l is None:
            alpha_l = np.linspace(0.0, 1., 100)

        # run cross-validation
        if n_jobs > 1:
            cvmse = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_fit_cv)(self,
                                 X,
                                 y,
                                 indices,
                                 PSF,
                                 n_splits,
                                 split_method,
                                 alpha,
                                 data_type=data_type,
                                 **kwargs) for alpha in alpha_l)
        else:
            cvmse = [
                _fit_cv(self,
                        X,
                        y,
                        indices,
                        PSF,
                        n_splits,
                        split_method,
                        alpha,
                        data_type=data_type,
                        **kwargs) for alpha in alpha_l
            ]

        cvmse = np.stack(cvmse)
        cvmse = np.hstack((alpha_l[:, None], cvmse))

        return cvmse

    def predict(self,
                X=None,
                indices=None,
                W=None,
                mean_factor=False,
                data_type="panel",
                label_ind=False):
        """wrapper around different data type predict methods
        Parameters
        ----------
        X :  numpy array or pandas DataFrame, optional
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.
            If given as a DataFrame, we assume that it contains a mutliindex
            mapping to each entity-time pair
            If None we use the values associated with the current model
        indices : numpy array, optional
            array containing the panel indices.  Should consist of two
            columns:
            - Column 1: entity id (i)
            - Column 2: time index (t)
            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
            If None we use the values associated with the current model
        W : numpy array, optional
            portfolio weight matrix of dimension (L, L, T)
            If None, we use the values associated with the current model
        mean_factor: boolean
            If true, the estimated factors are averaged in the time-series
            before prediction.
        data_type : str
            label for data-type used for prediction, one of the following:
            1. panel
            Uses the untransformed X and y for the estimation.
            2. portfolio
            Uses a matrix of characteristic weighted portfolios (Q)
            as well as a matrix of weights (W) and count of non-missing
            observations for each time period (val_obs) for the estimation.
            See _build_portfolio for details on how these variables are formed
            from the initial X and y.
        label_ind : bool
            whether to apply the indices to fitted values and return
            pandas Series
        Returns
        -------
        numpy array or pandas DataFrame/Series
            The exact value returned depends on two things:
            1. The data_type
                a. If panel data_type is specified, this will be a series of
                values for the panel ys
                b. If portfolio data_type is specified, this will a matrix
                of predicted char formed portfolio Qs
            2. label_ind
                If label_ind is True, we return pandas variants of the
                predicted values.  If not, we return the underlying
                numpy arrays.
        """

        if data_type == "panel":

            if X is None:
                X, indices, metad = self.X, self.indices, self.metad
            else:
                X, y, indices, metad = _prep_input(X, None, indices)
            N, L, T = metad["N"], metad["L"], metad["T"]

            pred = self.predict_panel(X, indices, T, mean_factor)

            if label_ind:
                pred = pd.DataFrame(pred, columns=["yhat"])
                ind = pd.DataFrame(indices, columns=["ids", "dates"])
                pred = pd.concat([ind, pred]).set_index(["ids", "dates"])

        elif data_type == "portfolio":

            if W is not None:
                L = W.shape[0]
                T = W.shape[2]
            elif X is not None:
                X, y, indices, metad = _prep_input(X, None, indices)
                Q, W, val_obs = _build_portfolio(X, None, indices, metad)
                L = W.shape[0]
                T = W.shape[2]
            elif hasattr(self, "W"):
                W = self.W
                L = W.shape[0]
                T = W.shape[2]
            else:
                X, indices, metad = self.X, self.indices, self.metad
                N, L, T = metad["N"], metad["L"], metad["T"]
                Q, W, val_obs = _build_portfolio(X, None, indices, metad)

            pred = self.predict_portfolio(W, L, T, mean_factor)

            if label_ind:
                pred = pd.DataFrame(pred,
                                    index=metad["chars"],
                                    columns=metad["dates"])

        else:
            raise ValueError("Unsupported data_type: %s" % data_type)

        return pred

    def predict_panel(self, X, indices, T, mean_factor=False):
        """
        Predicts fitted values for a previously fitted regressor + panel data
        Parameters
        ----------
        X :  numpy array
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.
        indices : numpy array
            array containing the panel indices.  Should consist of two
            columns:
            - Column 1: entity id (i)
            - Column 2: time index (t)
            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
        T : scalar
            number of time periods in X
        mean_factor: boolean
            If true, the estimated factors are averaged in the time-series
            before prediction.
        Returns
        -------
        ypred : numpy array
            The length of the returned array matches the length of data.
            A nan will be returned if there is missing chars information.
        """

        mean_Factors = np.mean(self.Factors, axis=1).reshape((-1, 1))

        if mean_factor:
            ypred = np.squeeze(X.dot(self.Gamma)\
                .dot(mean_Factors))
        elif T != self.Factors.shape[1]:
            raise ValueError("If mean_factor isn't used date shape must align\
                              with Factors shape")
        else:
            ypred = np.full((X.shape[0]), np.nan)
            for t in range(T):
                ix = (indices[:, 1] == t)
                ypred[ix] = np.squeeze(X[ix, :].dot(self.Gamma)\
                    .dot(self.Factors[:, t]))
        return ypred

    def predict_portfolio(self, W, L, T, mean_factor=False):
        """
        Predicts fitted values for a previously fitted regressor + portfolios
        Parameters
        ----------
        W : numpy array
            portfolio weight matrix of dimension (L, L, T)
        L : scalar
            number of characteristics
        T : scalar
            number of time periods
        mean_factor: boolean
            If true, the estimated factors are averaged in the time-series
            before prediction.
        Returns
        -------
        Qpred : numpy array
            Same dimensions as a char formed portfolios (Q)
        """

        # Compute goodness of fit measures, portfolio level
        Num_tot, Denom_tot, Num_pred, Denom_pred = 0, 0, 0, 0

        if not mean_factor and T != self.Factors.shape[1]:
            raise ValueError("If mean_factor isn't used date shape must align\
                              with Factors shape")

        if mean_factor:
            mean_Factors = np.mean(self.Factors, axis=1).reshape((-1, 1))

        Qpred = np.full((L, T), np.nan)
        for t in range(T):

            if mean_factor:
                qpred = W[:, :, t].dot(self.Gamma).dot(mean_Factors)
                qpred = np.squeeze(qpred)
                Qpred[:, t] = qpred
            else:
                qpred = W[:, :, t].dot(self.Gamma)\
                    .dot(self.Factors[:, t])
                Qpred[:, t] = qpred

        return Qpred

    def score(self,
              X,
              y=None,
              indices=None,
              mean_factor=False,
              data_type="panel"):
        """generate R^2
        Parameters
        ----------
        X :  numpy array or pandas DataFrame
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.
            If given as a DataFrame, we assume that it contains a mutliindex
            mapping to each entity-time pair
            If None we use the values associated with the current model
        y : numpy array or pandas Series, optional
            dependent variable where indices correspond to those in X
            If given as a Series, we assume that it contains a mutliindex
            mapping to each entity-time pair
        indices : numpy array, optional
            array containing the panel indices.  Should consist of two
            columns:
            - Column 1: entity id (i)
            - Column 2: time index (t)
            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
            If None we use the values associated with the current model
        mean_factor: boolean
            If true, the estimated factors are averaged in the time-series
            before prediction.
        data_type : str
            label for data-type used for prediction, one of the following:
            1. panel
            Uses the untransformed X and y for the estimation.
            2. portfolio
            Uses a matrix of characteristic weighted portfolios (Q)
            as well as a matrix of weights (W) and count of non-missing
            observations for each time period (val_obs) for the estimation.
            See _build_portfolio for details on how these variables are formed
            from the initial X and y.
        label_ind : bool
            whether to apply the indices to fitted values and return
            pandas Series
        Returns
        -------
        r2 : scalar
            summary of model performance
        """

        if data_type == "panel":

            X, y, indices, metad = _prep_input(X, y, indices)
            if y is None:
                y = self.y

            yhat = self.predict(X=X,
                                indices=indices,
                                mean_factor=mean_factor,
                                data_type="panel")

            return r2_score(y, yhat)

        elif data_type == "portfolio":

            X, y, indices, metad = _prep_input(X, y, indices)
            if y is None:
                y = self.y
            Q, W, val_obs = _build_portfolio(X, y, indices, metad)

            Qhat = self.predict(W=W,
                                mean_factor=mean_factor,
                                data_type="portfolio")
            return r2_score(Q, Qhat)

        else:
            return ValueError("Unsupported data_type: %s" % data_type)

    def BS_Walpha(self, ndraws=1000, n_jobs=1, backend='loky'):
        """
        Bootstrap inference on the hypothesis Gamma_alpha = 0
        Parameters
        ----------
        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed
        backend : optional
            Value is either 'loky' or 'multiprocessing'
        n_jobs  : integer
            Number of workers to be used. If -1, all available workers are
            used.
        Returns
        -------
        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        if self.alpha > 0.:
            raise ValueError("Bootstrap currently not supported for\
                              regularized estimation.")

        if not self.intercept:
            raise ValueError('Need to fit model with intercept first.')

        # fail if model isn't estimated
        if not hasattr(self, "Q"):
            raise ValueError("Bootstrap can only be run on fitted model.")

        N, L, T = self.metad["N"], self.metad["L"], self.metad["T"]

        # Compute Walpha
        Walpha = self.Gamma[:, -1].T.dot(self.Gamma[:, -1])

        # Compute residuals
        d = np.full((L, T), np.nan)

        for t_i, t in enumerate(self.metad["dates"]):
            d[:, t_i] = self.Q[:, t_i]-self.W[:, :, t_i].dot(self.Gamma)\
                .dot(self.Factors[:, t_i])

        print("Starting Bootstrap...")
        Walpha_b = Parallel(n_jobs=n_jobs, backend=backend,
                            verbose=10)(delayed(_BS_Walpha_sub)(self, n, d)
                                        for n in range(ndraws))
        print("Done!")

        # print(Walpha_b, Walpha)
        pval = np.sum(Walpha_b > Walpha) / ndraws
        return pval

    def BS_Wbeta(self, l, ndraws=1000, n_jobs=1, backend='loky'):
        """
        Test of instrument significance.
        Bootstrap inference on the hypothesis  l-th column of Gamma_beta = 0.
        Parameters
        ----------
        l   : integer
            Position of the characteristics for which the bootstrap is to be
            carried out. For example, if there are 10 characteristics, l is in
            the range 0 to 9 (left-/right-inclusive).
        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed
        n_jobs  : integer
            Number of workers to be used for multiprocessing.
            If -1, all available Workers are used.
        backend : optional
        Returns
        -------
        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        if self.alpha > 0.:
            raise ValueError("Bootstrap currently not supported for\
                              regularized estimation.")

        if self.PSFcase:
            raise ValueError('Need to fit model without intercept first.')

        # fail if model isn't estimated
        if not hasattr(self, "Q"):
            raise ValueError("Bootstrap can only be run on fitted model.")

        N, L, T = self.metad["N"], self.metad["L"], self.metad["T"]

        # Compute Wbeta_l if l-th characteristics is set to zero
        Wbeta_l = np.sum(np.square(self.Gamma[l, :]))

        # Compute residuals
        d = np.full((L, T), np.nan)
        for t_i, t in enumerate(self.metad["dates"]):
            d[:, t_i] = self.Q[:, t_i]-self.W[:, :, t_i].dot(self.Gamma)\
                .dot(self.Factors[:, t_i])

        print("Starting Bootstrap...")
        Wbeta_l_b = Parallel(n_jobs=n_jobs, backend=backend,
                             verbose=10)(delayed(_BS_Wbeta_sub)(self, n, d, l)
                                         for n in range(ndraws))
        print("Done!")

        pval = np.sum(Wbeta_l_b > Wbeta_l) / ndraws

        return pval

    def BS_Wdelta(self, ndraws=1000, n_jobs=1, backend='loky'):
        """
        Test of PSF significance.
        Bootstrap inference on the hypothesis Gamma_delta = 0. Assumes that only one PSF is used and no intercept is in use.
        Parameters
        ----------
        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed
        n_jobs  : integer
            Number of workers to be used for multiprocessing.
            If -1, all available Workers are used.
        backend : optional
        Returns
        -------
        pval : float
            P-value from the hypothesis test H0: Gamma_delta=0
        """

        if self.alpha > 0.:
            raise ValueError("Bootstrap currently not supported for\
                              regularized estimation.")

        if self.intercept:
            raise ValueError('Need to fit model without intercept first.')

        if not self.PSFcase:
            raise ValueError('Need to fit model with one PSF first.')

        K_PSF, _ = np.shape(self.PSF)

        if K_PSF > 1:
            raise ValueError('Not implemented for more than one PSF yet.')

        # fail if model isn't estimated
        if not hasattr(self, "Q"):
            raise ValueError("Bootstrap can only be run on fitted model.")

        N, L, T = self.metad["N"], self.metad["L"], self.metad["T"]

        # Compute Wdelta if Gamma_delta is set to zero
        Wdelta = self.Gamma[:, -1].T.dot(self.Gamma[:, -1])
        # Compute residuals
        d = np.full((L, T), np.nan)
        for t_i, t in enumerate(self.metad["dates"]):
            d[:, t_i] = self.Q[:, t_i]-self.W[:, :, t_i].dot(self.Gamma)\
                .dot(self.Factors[:, t_i])

        print("Starting Bootstrap...")
        Wdelta_b = Parallel(n_jobs=n_jobs, backend=backend,
                            verbose=10)(delayed(_BS_Wdelta_sub)(self, n, d)
                                        for n in range(ndraws))
        print("Done!")

        #print(Wdelta_b, Wdelta)
        pval = np.sum(Wdelta_b > Wdelta) / ndraws

        return pval

    def predictOOS(self, X=None, y=None, indices=None, mean_factor=False):
        """
        Predicts time t+1 observation using an out-of-sample design.
        Parameters
        ----------
        X :  numpy array
            X of stacked data. Each row corresponds to an observation
            (i,t) where i denotes the entity index and t denotes
            the time index. All data must correspond to time t, i.e. all
            observations occur on the same date.
            If an observation contains missing data NaN will be returned.
            Note that the number of characteristics (L) passed,
            has to match the number of characteristics used when fitting
            the regressor.
            The columns of the panel are organized in the following order:
            - Column 1: entity id (i)
            - Column 2: time index (t)
            - Column 3 to column 3+L: characteristics.
        y : numpy array
            dependent variable where indices correspond to those in X
        indices : numpy array
            array containing the panel indices.  Should consist of two
            columns:
            - Column 1: entity id (i)
            - Column 2: time index (t)
            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
        mean_factor: boolean
            If true, the estimated factors are averaged in the time-series
            before prediction.
        Returns
        -------
        ypred : numpy array
            The length of the returned array matches the
            the length of data. A nan will be returned if there is missing
            characteristics information.
        """

        X, y, indices, metad = _prep_input(X, y, indices)
        N, L, T = metad["N"], metad["L"], metad["T"]
        ypred = np.full((N), np.nan)

        # Compute realized factor returns
        Numer = self.Gamma.T.dot(X.T).dot(y)
        Denom = self.Gamma.T.dot(X.T).dot(X).dot(self.Gamma)
        Factor_OOS = np.linalg.solve(Denom, Numer.reshape((-1, 1)))

        if mean_factor:
            ypred = np.squeeze(X.dot(self.Gamma)\
                    .dot(np.mean(self.Factors, axis=1).reshape((-1, 1))))
        else:
            ypred = np.diag(X.dot(self.Gamma).dot(Factor_OOS))

        return ypred

    def _fit_ipca(self,
                  X=None,
                  y=None,
                  indices=None,
                  PSF=None,
                  Q=None,
                  W=None,
                  val_obs=None,
                  Gamma=None,
                  Factors=None,
                  quiet=False,
                  data_type="portfolio",
                  **kwargs):
        """
        Fits the regressor to the data using alternating least squares
        Parameters
        ----------
        X :  numpy array, optional
            matrix of characteristics where each row corresponds to a
            entity-time pair in indices.  The number of characteristics
            (columns here) used as instruments is L.
        y : numpy array, optional
            dependent variable where indices correspond to those in X
        indices : numpy array, optional
            array containing the panel indices.  Should consist of two
            columns:
            - Column 1: entity id (i)
            - Column 2: time index (t)
            The panel may be unbalanced. The number of unique entities is
            n_samples, the number of unique dates is T, and the number of
            characteristics used as instruments is L.
        PSF : array-like of shape (M, T), optional
            pre-specified factors
        Q : array-like of shape (L,T), optional
            characteristics weighted portfolios
        W : array_like of shape (L, L,T), optional
            portfolio weights
        val_obs: array-like, optional
            matrix of dimension (T), containting the number of non missing
            observations at each point in time
        Gamma : numpy array, optional
            If provided, starting values for Gamma
        Factors : numpy array, optional
            If provided, starting values for Factors
        data_type : str
            label for method used when fitting ALS, should be one of:
            1. panel
            2. portfolio
        quiet   : optional, bool
            If true no text output will be produced
        Returns
        -------
        Gamma : array-like with dimensions (L, n_factors). If there
            are n_prespec many pre-specified factors in the model then the
            matrix returned is of dimension (L, (n_factors+M)).
            If an intercept is included in the model, its loadings are
            returned in the last column of Gamma.
        Factors : array_like with dimensions (n_factors, T). If
            pre-specified factors were passed the returned matrix is
            of dimension ((n_factors - M), T), corresponding to the
            n_factors - M many factors estimated on top of the pre-
            specified ones.
        """

        if data_type == "panel":
            ALS_inputs = (X, y, indices)
            ALS_fit = self._ALS_fit_panel
        elif data_type == "portfolio":
            ALS_inputs = (Q, W, val_obs)
            ALS_fit = self._ALS_fit_portfolio
        else:
            raise ValueError("Unsupported ALS method: %s" % data_type)

        # Initialize the Alternating Least Squares Procedure
        if Gamma is None or Factors is None:
            Gamma_Old, s, v = np.linalg.svd(Q)
            Gamma_Old = Gamma_Old[:, :self.n_factors_eff]
            s = s[:self.n_factors_eff]
            v = v[:self.n_factors_eff, :]
            Factor_Old = np.diag(s).dot(v)
        if Gamma is not None:
            Gamma_Old = Gamma
        if Factors is not None:
            Factors_Old = Factors

        # Estimation Step
        tol_current = 1

        iter = 0

        while ((iter <= self.max_iter) and (tol_current > self.iter_tol)):

            Gamma_New, Factor_New = ALS_fit(Gamma_Old,
                                            *ALS_inputs,
                                            PSF=PSF,
                                            **kwargs)

            if self.PSFcase:
                tol_current = np.max(np.abs(Gamma_New - Gamma_Old))
            else:
                tol_current_G = np.max(np.abs(Gamma_New - Gamma_Old))
                tol_current_F = np.max(np.abs(Factor_New - Factor_Old))
                tol_current = max(tol_current_G, tol_current_F)

            # Update factors and loadings
            Factor_Old, Gamma_Old = Factor_New, Gamma_New

            iter += 1
            if not quiet:
                print('Step', iter, '- Aggregate Update:', tol_current)

        if not quiet:
            print('-- Convergence Reached --')

        return Gamma_New, Factor_New

    def _ALS_fit_portfolio(self, Gamma_Old, Q, W, val_obs, PSF=None, **kwargs):
        """Alternating least squares procedure to fit params
        Runs using portfolio data as input
        The alternating least squares procedure switches back and forth
        between evaluating the first order conditions for Gamma_Beta, and the
        factors until convergence is reached. This function carries out one
        complete update procedure and will need to be called repeatedly using
        the updated Gamma's and factors as inputs.
        """

        T = self.metad["T"]

        if PSF is None:
            L, K = np.shape(Gamma_Old)
            Ktilde = K
        else:
            L, Ktilde = np.shape(Gamma_Old)
            K_PSF, _ = np.shape(PSF)
            K = Ktilde - K_PSF

        # ALS Step 1
        if K > 0:

            # case with no observed factors
            if PSF is None:
                if self.n_jobs > 1:
                    F_New = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                        delayed(_Ft_fit_portfolio)(Gamma_Old, W[:, :, t], Q[:,
                                                                            t])
                        for t in range(T))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t in range(T):
                        F_New[:,
                              t] = _Ft_fit_portfolio(Gamma_Old, W[:, :, t],
                                                     Q[:, t])

            # observed factors+latent factors case
            else:
                if self.n_jobs > 1:
                    F_New = Parallel(n_jobs=n_jobs, backend=backend)(
                        delayed(_Ft_fit_PSF_portfolio)
                        (Gamma_Old, W[:, :, t], Q[:, t], PSF[:, t], K, Ktilde)
                        for t in range(T))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t in range(T):
                        F_New[:, t] = _Ft_fit_PSF_portfolio(
                            Gamma_Old, W[:, :, t], Q[:, t], PSF[:, t], K,
                            Ktilde)

        else:
            F_New = None

        # ALS Step 2
        Gamma_New = _Gamma_fit_portfolio(F_New, Q, W, val_obs, PSF, L, K,
                                         Ktilde, T)

        # condition checks

        if K > 0:
            # Enforce Orthogonality in Gamma_alpha and Gamma_beta
            R1 = _numba_chol(Gamma_New[:, :K].T.dot(Gamma_New[:, :K])).T
            R2, _, _ = _numba_svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
            Gamma_New[:, :K] = _numba_lstsq(Gamma_New[:, :K].T,
                                            R1.T)[0].dot(R2)
            F_New = _numba_solve(R2, R1.dot(F_New))

            # Enforce sign convention for Gamma_Beta and F_New
            sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            sg[sg == 0] = 1
            Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            F_New = np.multiply(F_New, sg)

            if PSF is not None:
                Gamma_New[:,
                          K:] = (np.identity(Gamma_New.shape[0]) -
                                 Gamma_New[:, :K].dot(Gamma_New[:, :K].T)).dot(
                                     Gamma_New[:, K:])
                F_New += Gamma_New[:, :K].T.dot(Gamma_New[:, K:]).dot(PSF)

                sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
                sg[sg == 0] = 1
                Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
                F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New

    def _ALS_fit_panel(self, Gamma_Old, X, y, indices, PSF=None, **kwargs):
        """Alternating least squares procedure to fit params
        Runs using panel data as input
        The alternating least squares procedure switches back and forth
        between evaluating the first order conditions for Gamma_Beta, and the
        factors until convergence is reached. This function carries out one
        complete update procedure and will need to be called repeatedly using
        the updated Gamma's and factors as inputs.
        """

        T, dates = self.metad["T"], self.metad["dates"]

        if PSF is None:
            L, K = np.shape(Gamma_Old)
            Ktilde = K
        else:
            L, Ktilde = np.shape(Gamma_Old)
            K_PSF, _ = np.shape(PSF)
            K = Ktilde - K_PSF

        # prep T-ind for iteration
        Tind = [np.where(indices[:, 1] == t)[0] for t in range(T)]

        # ALS Step 1
        if K > 0:

            # case with no observed factors
            if PSF is None:
                if self.n_jobs > 1:
                    F_New = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                        delayed(_Ft_fit_panel)(Gamma_Old, X[tind, :], y[tind])
                        for t, tind in enumerate(Tind))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t, tind in enumerate(Tind):
                        F_New[:, t] = _Ft_fit_panel(Gamma_Old, X[tind, :],
                                                    y[tind])

            # observed factors+latent factors case
            else:
                if self.n_jobs > 1:
                    F_New = Parallel(n_jobs=n_jobs, backend=backend)(
                        delayed(_Ft_fit_PSF_panel)(Gamma_Old, X[
                            tind, :], y[tind], PSF[:, t], K, Ktilde)
                        for t, tind in enumerate(Tind))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t, tind in enumerate(Tind):
                        F_New[:,
                              t] = _Ft_fit_PSF_panel(Gamma_Old, X[tind, :],
                                                     y[tind], PSF[:, t], K,
                                                     Ktilde)

        else:
            F_New = None

        # ALS Step 2
        Gamma_New = _Gamma_fit_panel(F_New, X, y, indices, PSF, L, Ktilde,
                                     self.alpha, self.l1_ratio, **kwargs)

        # condition checks

        if K > 0:
            # Enforce Orthogonality in Gamma_alpha and Gamma_beta
            R1 = _numba_chol(Gamma_New[:, :K].T.dot(Gamma_New[:, :K])).T
            R2, _, _ = _numba_svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
            Gamma_New[:, :K] = _numba_lstsq(Gamma_New[:, :K].T,
                                            R1.T)[0].dot(R2)
            F_New = _numba_solve(R2, R1.dot(F_New))

            # Enforce sign convention for Gamma_Beta and F_New
            sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            sg[sg == 0] = 1
            Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            F_New = np.multiply(F_New, sg)

            if PSF is not None:
                Gamma_New[:,
                          K:] = (np.identity(Gamma_New.shape[0]) -
                                 Gamma_New[:, :K].dot(Gamma_New[:, :K].T)).dot(
                                     Gamma_New[:, K:])
                F_New += Gamma_New[:, :K].T.dot(Gamma_New[:, K:]).dot(PSF)

                sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
                sg[sg == 0] = 1
                Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
                F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New


def _prep_input(X, y=None, indices=None):
    """handle mapping from different inputs type to consistent internal data
    Parameters
    ----------
    X :  numpy array or pandas DataFrame
        matrix of characteristics where each row corresponds to a
        entity-time pair in indices.  The number of characteristics
        (columns here) used as instruments is L.
        If given as a DataFrame, we assume that it contains a mutliindex
        mapping to each entity-time pair
    y : numpy array or pandas Series, optional
        dependent variable where indices correspond to those in X
        If given as a Series, we assume that it contains a mutliindex
        mapping to each entity-time pair
    indices : numpy array or pandas MultiIndex, optional
        array containing the panel indices.  Should consist of two
        columns:
        - Column 1: entity id (i)
        - Column 2: time index (t)
        The panel may be unbalanced. The number of unique entities is
        n_samples, the number of unique dates is T, and the number of
        characteristics used as instruments is L.
    Returns
    -------
    X :  numpy array
        matrix of characteristics where each row corresponds to a
        entity-time pair in indices.  The number of characteristics
        (columns here) used as instruments is L.
    y : numpy array
        dependent variable where indices correspond to those in X
    indices : numpy array
        array containing the panel indices.  Should consist of two
        columns:
        - Column 1: entity id (i)
        - Column 2: time index (t)
        The panel may be unbalanced. The number of unique entities is
        n_samples, the number of unique dates is T, and the number of
        characteristics used as instruments is L.
    metad : dict
        contains metadata on inputs:
        dates : array-like
            unique dates in panel
        ids : array-like
            unique ids in panel
        chars : array-like
            labels for X chars/columns
        T : scalar
            number of time periods
        N : scalar
            number of ids
        L : scalar
            total number of characteristics
    """

    # Check panel input
    if X is None:
        raise ValueError('Must pass panel input data.')
    else:
        # remove panel rows containing missing obs
        non_nan_ind = ~np.any(np.isnan(X), axis=1)
        X = X[non_nan_ind]
        if y is not None:
            y = y[non_nan_ind]

    # if data-frames passed, break out indices from data
    if isinstance(X, pd.DataFrame) and not isinstance(y, pd.Series):
        indices = X.index
        chars = X.columns
        X = X.values
    elif not isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        indices = y.index
        y = y.values
        chars = np.arange(X.shape[1])
    elif isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        Xind = X.index
        chars = X.columns
        yind = y.index
        X = X.values
        y = y.values
        if not np.array_equal(Xind, yind):
            raise ValueError("If indices are provided with both X and y\
                              they be the same")
        indices = Xind
    else:
        chars = np.arange(X.shape[1])

    if indices is None:
        raise ValueError("entity-time indices must be provided either\
                          separately or as a MultiIndex with X/y")

    # extract numpy array and labels from multiindex
    if isinstance(indices, pd.MultiIndex):
        indices = indices.to_frame().values
    ids = np.unique(indices[:, 0])
    dates = np.unique(indices[:, 1])
    indices[:, 0] = np.unique(indices[:, 0], return_inverse=True)[1]
    indices[:, 1] = np.unique(indices[:, 1], return_inverse=True)[1]

    # init data dimensions
    T = np.size(dates, axis=0)
    N = np.size(ids, axis=0)
    L = np.size(chars, axis=0)

    # prep metadata
    metad = {}
    metad["dates"] = dates
    metad["ids"] = ids
    metad["chars"] = chars
    metad["T"] = T
    metad["N"] = N
    metad["L"] = L

    return X, y, indices, metad


def _build_portfolio(X, y, indices, metad):
    """ Converts a stacked panel of data where each row corresponds to an
    observation (i, t) into a tensor of dimensions (N, L, T) where N is the
    number of unique entities, L is the number of characteristics and T is
    the number of unique dates
    Parameters
    ----------
    X :  numpy array
        matrix of characteristics where each row corresponds to a
        entity-time pair in indices.  The number of characteristics
        (columns here) used as instruments is L.
    y : numpy array, optional
        dependent variable where indices correspond to those in X
        If None, we just build the portfolio info for ind vars
    indices : numpy array
        array containing the panel indices.  Should consist of two
        columns:
        - Column 1: entity id (i)
        - Column 2: time index (t)
        The panel may be unbalanced. The number of unique entities is
        n_samples, the number of unique dates is T, and the number of
        characteristics used as instruments is L.
    metad : dict
        dictionary containing metadata
    Returns
    -------
    Q: array-like, optional
        matrix of dimensions (L, T), containing the characteristics
        weighted portfolios
    W: array-like, optional
        matrix of dimension (L, L, T)
    val_obs: array-like
        matrix of dimension (T), containting the number of non missing
        observations at each point in time
    """

    N, L, T = metad["N"], metad["L"], metad["T"]

    print('The panel dimensions are:')
    print('n_samples:', N, ', L:', L, ', T:', T)

    bar = progressbar.ProgressBar(maxval=T,
                                  widgets=[
                                      progressbar.Bar('=', '[', ']'), ' ',
                                      progressbar.Percentage()
                                  ])
    bar.start()

    # init portfolio outputs based on input type
    if y is not None:
        Q = np.full((L, T), np.nan)
    else:
        Q = None
    W = np.full((L, L, T), np.nan)
    val_obs = np.full((T), np.nan)

    if y is None:
        for t in range(T):
            ixt = (indices[:, 1] == t)
            val_obs[t] = np.sum(ixt)
            W[:, :, t] = X[ixt, :].T.dot(X[ixt, :]) / val_obs[t]
            bar.update(t)

    else:
        for t in range(T):
            ixt = (indices[:, 1] == t)
            val_obs[t] = np.sum(ixt)
            Q[:, t] = X[ixt, :].T.dot(y[ixt]) / val_obs[t]
            W[:, :, t] = X[ixt, :].T.dot(X[ixt, :]) / val_obs[t]
            bar.update(t)

    bar.finish()

    # return portfolio data
    return Q, W, val_obs


def _Ft_fit_portfolio(Gamma_Old, W_t, Q_t):
    """helper func to parallelize F ALS fit"""

    m1 = Gamma_Old.T.dot(W_t).dot(Gamma_Old)
    m2 = Gamma_Old.T.dot(Q_t)

    return np.squeeze(_numba_solve(m1, m2.reshape((-1, 1))))


def _Ft_fit_PSF_portfolio(Gamma_Old, W_t, Q_t, PSF_t, K, Ktilde):
    """helper func to parallelize F ALS fit with observed factors"""

    m1 = Gamma_Old[:, :K].T.dot(W_t).dot(Gamma_Old[:, :K])
    m2 = Gamma_Old[:, :K].T.dot(Q_t)
    m2 -= Gamma_Old[:, :K].T.dot(W_t).dot(Gamma_Old[:, K:Ktilde]).dot(PSF_t)

    return np.squeeze(_numba_solve(m1, m2.reshape((-1, 1))))


def _Ft_fit_panel(Gamma_Old, X_t, y_t):
    """fits F_t using panel data"""

    exog_t = X_t.dot(Gamma_Old)
    Ft = _numba_lstsq(exog_t, y_t)[0]

    return Ft


def _Ft_fit_PSF_panel(Gamma_Old, X_t, y_t, PSF_t, K, Ktilde):
    """fits F_t using panel data with PSF"""

    exog_t = X_t.dot(Gamma_Old)
    y_t_resid = y_t - exog_t[:, K:Ktilde].dot(PSF_t)
    Ft = _numba_lstsq(exog_t[:, :K], y_t_resid)[0]

    return Ft


def _Gamma_fit_portfolio(F_New, Q, W, val_obs, PSF, L, K, Ktilde, T):
    """helper function for fitting gamma without panel"""

    Numer = _numba_full((L * Ktilde, 1), 0.0)
    Denom = _numba_full((L * Ktilde, L * Ktilde), 0.0)

    # no observed factors
    if PSF is None:
        for t in range(T):

            Numer += _numba_kron(Q[:, t].reshape((-1, 1)),
                                 F_New[:, t].reshape((-1, 1)))\
                                 * val_obs[t]
            Denom += _numba_kron(W[:, :, t],
                                 F_New[:, t].reshape((-1, 1))
                                 .dot(F_New[:, t].reshape((1, -1)))) \
                                 * val_obs[t]

    # observed+latent factors
    elif K > 0:
        for t in range(T):
            Numer += _numba_kron(Q[:, t].reshape((-1, 1)),
                                 np.vstack(
                                 (F_New[:, t].reshape((-1, 1)),
                                 PSF[:, t].reshape((-1, 1)))))\
                                 * val_obs[t]
            Denom_temp = np.vstack((F_New[:, t].reshape(
                (-1, 1)), PSF[:, t].reshape((-1, 1))))
            Denom += _numba_kron(W[:, :, t],
                                 Denom_temp.dot(Denom_temp.T) * val_obs[t])

    # only observed factors
    else:
        for t in range(T):
            Numer += _numba_kron(Q[:, t].reshape((-1, 1)),
                                 PSF[:, t].reshape((-1, 1)))\
                                 * val_obs[t]
            Denom += _numba_kron(W[:, :, t],
                                 PSF[:, t].reshape((-1, 1))
                                 .dot(PSF[:, t].reshape((-1, 1)).T))\
                                 * val_obs[t]

    Gamma_New = _numba_solve(Denom, Numer).reshape((L, Ktilde))

    return Gamma_New


def _Gamma_fit_panel(F_New, X, y, indices, PSF, L, Ktilde, alpha, l1_ratio,
                     **kwargs):
    """helper function for estimating vectorized Gamma with panel"""

    # join observed factors with latent factors and map to panel
    if PSF is None:
        F = F_New
    else:
        if F_New is None:
            F = PSF
        else:
            F = np.vstack((F_New, PSF))
    F = F[:, indices[:, 1]]

    # interact factors and characteristics
    ZkF = np.hstack([F[k, :, None] * X for k in range(Ktilde)])

    # elastic net fit
    if alpha:
        mod = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        mod.fit(ZkF, y)
        gamma = mod.coef_

    # OLS fit
    else:
        gamma = _numba_lstsq(ZkF, y)[0]

    gamma = gamma.reshape((Ktilde, L)).T

    return gamma


def _fit_cv(model, X, y, indices, PSF, n_splits, split_method, alpha,
            **kwargs):
    """inner function for fit_path doing CV
    Parameters
    ----------
    model : IPCA model instance
        contains the params
    X :  numpy array
        matrix of characteristics where each row corresponds to a
        entity-time pair in indices.  The number of characteristics
        (columns here) used as instruments is L.
    y : numpy array
        dependent variable where indices correspond to those in X
    indices : numpy array
        array containing the panel indices.  Should consist of two
        columns:
        - Column 1: entity id (i)
        - Column 2: time index (t)
        The panel may be unbalanced. The number of unique entities is
        n_samples, the number of unique dates is T, and the number of
        characteristics used as instruments is L.
    PSF : numpy array, optional
        Set of pre-specified factors as matrix of dimension (M, T)
    n_splits : scalar
        number of CV partitions
    split_method : sklearn cross-validation generator factory
        method to generate CV partitions
    alpha : scalar
        regularizing constant for current step in elastic-net path
    Returns
    -------
    mse : array
        array of MSEs for each CV partition
    Notes
    -----
    Groups are defined by 'ids'
    """

    # build iterator
    mse_l = []
    split = split_method(n_splits=n_splits)

    full_tind = np.unique(indices[:, 1])

    for train, test in split.split(indices, groups=indices[:, 0]):

        # build partitioned model
        train_X = X[train, :]
        train_y = y[train]
        train_indices = indices[train, :]
        test_X = X[test, :]
        test_y = y[test]
        test_indices = indices[test, :]
        if PSF is None:
            train_PSF = None
        else:
            train_tind = np.unique(train_indices[:, 1])
            train_tind = np.where(np.isin(full_tind, train_tind))[0]
            train_PSF = PSF[:, train_tind]

        # init new training model
        params = model.get_params()
        params["alpha"] = alpha
        train_IPCA = InstrumentedPCA(**params)
        train_IPCA = train_IPCA.fit(train_X, train_y, train_indices, train_PSF,
                                    **kwargs)

        # get MSE
        test_pred = train_IPCA.predict(test_X, test_indices, mean_factor=True)
        mse = np.sum(np.square(test_y - test_pred))
        mse /= test_pred.shape[0]
        mse_l.append(mse)

    return np.array(mse_l)


def _BS_Walpha_sub(model, n, d):
    L, T = model.metad["L"], model.metad["T"]
    Q_b = np.full((L, T), np.nan)
    np.random.seed(n)

    # Re-estimate unrestricted model
    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                Q_b[:, t] = model.W[:, :, t].dot(model.Gamma[:, :-1])\
                    .dot(model.Factors[:-1, t]) + d_temp
            Gamma, Factors = model._fit_ipca(Q=Q_b,
                                             W=model.W,
                                             val_obs=model.val_obs,
                                             PSF=model.PSF,
                                             quiet=True,
                                             data_type="portfolio")
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    # Compute and store Walpha_b
    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return Walpha_b


def _BS_Wbeta_sub(model, n, d, l):
    L, T = model.metad["L"], model.metad["T"]
    Q_b = np.full((L, T), np.nan)
    np.random.seed(n)
    #Modify Gamma_beta such that its l-th row is zero
    Gamma_beta_l = np.copy(model.Gamma)
    Gamma_beta_l[l, :] = 0

    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                Q_b[:, t] = model.W[:, :, t].dot(Gamma_beta_l)\
                    .dot(model.Factors[:, t]) + d_temp
            Gamma, Factors = model._fit_ipca(Q=Q_b,
                                             W=model.W,
                                             val_obs=model.val_obs,
                                             PSF=model.PSF,
                                             quiet=True,
                                             data_type="portfolio")

        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    # Compute and store Walpha_b
    Wbeta_l_b = Gamma[l, :].dot(Gamma[l, :].T)
    Wbeta_l_b = np.trace(Wbeta_l_b)
    return Wbeta_l_b


def _BS_Wdelta_sub(model, n, d):
    L, T = model.metad["L"], model.metad["T"]
    Q_b = np.full((L, T), np.nan)
    np.random.seed(n)

    #Modify Gamma_delta such that its last row for the PSF is zero
    Gamma_delta = np.copy(model.Gamma)
    Gamma_delta[:, -1] = 0

    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                Q_b[:, t] = model.W[:, :, t].dot(Gamma_delta)\
                    .dot(model.Factors[:, t]) + d_temp
            Gamma, Factors = model._fit_ipca(Q=Q_b,
                                             W=model.W,
                                             val_obs=model.val_obs,
                                             PSF=model.PSF,
                                             quiet=True,
                                             data_type="portfolio")

        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                               Observation discarded.")
            pass

    # Compute and store Wdelta_b
    Wdelta_b = Gamma[:, -1].T.dot(Gamma[:, -1])
    return Wdelta_b


@jit(nopython=True)
def _numba_solve(m1, m2):
    return np.linalg.solve(np.ascontiguousarray(m1), np.ascontiguousarray(m2))


@jit(nopython=True)
def _numba_lstsq(m1, m2):
    return np.linalg.lstsq(np.ascontiguousarray(m1), np.ascontiguousarray(m2))


@jit(nopython=True)
def _numba_kron(m1, m2):
    return np.kron(np.ascontiguousarray(m1), np.ascontiguousarray(m2))


@jit(nopython=True)
def _numba_chol(m1):
    return np.linalg.cholesky(np.ascontiguousarray(m1))


@jit(nopython=True)
def _numba_svd(m1):
    return np.linalg.svd(np.ascontiguousarray(m1))


@jit(nopython=True)
def _numba_full(m1, m2):
    return np.full(m1, m2)
