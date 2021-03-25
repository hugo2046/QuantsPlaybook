'''
Author: guofei9987
Date: 2020-08-29 13:08:41
LastEditTime: 2021-03-08 19:04:41
LastEditors: Please set LastEditors
Description: 差分进化算法

from:https://github.com/guofei9987/scikit-opt/blob/master/sko/DE.py
'''
import numpy as np
from functools import lru_cache
from abc import ABCMeta, abstractmethod


def func_transformer(func):
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible
    :param func:
    :return:
    '''

    prefered_function_format = '''
    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    '''

    is_vector = getattr(func, 'is_vector', False)
    is_parallel = getattr(func, 'is_parallel', False)
    is_cached = getattr(func, 'is_cached', False)

    if is_cached:
        @lru_cache(maxsize=None)
        def func_cached(X):
            return func
        func = func_cached

    if is_vector:
        return func

    if is_parallel:
        from multiprocessing.dummy import Pool

        pool = Pool()

        def func_transformed(X):
            return np.array(pool.map(func, X))

        return func_transformed
    else:
        if func.__code__.co_argcount == 1:
            def func_transformed(X):
                return np.array([func(x) for x in X])

            return func_transformed
        elif func.__code__.co_argcount > 1:

            def func_transformed(X):
                return np.array([func(*tuple(x)) for x in X])

            return func_transformed

    raise ValueError('''
    object function error,
    function should be like this:
    ''' + prefered_function_format)


class SkoBase(metaclass=ABCMeta):
    def register(self, operator_name, operator, *args, **kwargs):
        '''
        regeister udf to the class
        :param operator_name: string
        :param operator: a function, operator itself
        :param args: arg of operator
        :param kwargs: kwargs of operator
        :return:
        '''

        def operator_wapper(*wrapper_args):
            return operator(*(wrapper_args + args), **kwargs)

        setattr(self, operator_name, types.MethodType(operator_wapper, self))
        return self

    def fit(self, *args, **kwargs):
        warnings.warn(
            '.fit() will be deprecated in the future. use .run() instead.', DeprecationWarning)
        return self.run(*args, **kwargs)


class GeneticAlgorithmBase(SkoBase, metaclass=ABCMeta):
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        self.func = func_transformer(func)
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        # a list of equal functions with ceq[i] = 0
        self.constraint_eq = list(constraint_eq)
        # a list of unequal constraint functions with c[i] <= 0
        self.constraint_ueq = list(constraint_ueq)

        self.Chrom = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.Y = None
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None

    @abstractmethod
    def chrom2x(self, Chrom):
        pass

    def x2y(self):
        self.Y_raw = self.func(self.X)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array(
                [np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array(
                [np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    @abstractmethod
    def ranking(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

    fit = run


class DE(GeneticAlgorithmBase):
    def __init__(self, func, n_dim, F=0.5,
                 size_pop=50, max_iter=200, prob_mut=0.3,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut,
                         constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)

        self.F = F
        self.V, self.U = None, None
        self.lb, self.ub = np.array(
            lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.crtbp()

    def crtbp(self):
        # create the population
        self.X = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        return self.X

    def chrom2x(self, Chrom):
        pass

    def ranking(self):
        pass
    
    @staticmethod
    def _check_r(random_idx):
        
        '''r1,r2,r3不应当为0'''
        
        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        
        while np.all(r3 == r2) & all(r3 == r1) & all(r2 == r1):
            random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))
            r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
            
        return r1, r2, r3
    
    def mutation(self):
        '''
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        '''
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal ==> _check_r
        random_idx = np.random.randint(
            0, self.size_pop, size=(self.size_pop, 3))
        
        r1, r2, r3 = self._check_r(random_idx)
        #r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]

        # 这里F用固定值，为了防止早熟，可以换成自适应值
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

        # the lower & upper bound still works in mutation
        mask = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)
        return self.V

    def crossover(self):
        '''
        if rand < prob_crossover, use V, else use X
        '''
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        '''
        greedy selection
        '''
        X = self.X.copy()
        f_X = self.x2y().copy()
        self.X = U = self.U
        f_U = self.x2y()

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)
        return self.X

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.selection()

            # record the best ones
            generation_best_index = self.Y.argmin()
            self.generation_best_X.append(
                self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y



