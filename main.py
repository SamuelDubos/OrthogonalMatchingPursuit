"""
author: @Samuel_Dubos
"""

from numpy import argmax, array, cos, dot, diag, divide, ones, pi
from matplotlib.pyplot import figure, plot, show
from sklearn import preprocessing
from numpy.linalg import norm
from pandas import read_csv
import fsspec

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.core.variable import Real
from pymoo.optimize import minimize


class SingleObjectiveOptimizationProblem(ElementwiseProblem):

    def __init__(self, obs_signal, dictionary, n_var,
                 t_clip=None, t_max=None, pos_matrix=None, neg_matrix=None, constraints=False):

        self.obs_signal = obs_signal
        self.dictionary = dictionary
        self.n_var = n_var
        self.t_clip = t_clip
        self.t_max = t_max
        self.pos_matrix = pos_matrix
        self.neg_matrix = neg_matrix
        self.constraints = constraints

        real_number = Real(bounds=(0, 1e2))
        arr = [_ for _ in range(self.n_var)]
        typ = [real_number for _ in range(self.n_var)]
        var_vector = dict(zip(arr, typ))
        vars = var_vector
        self.n_ieq_constr = 0
        if constraints:
            self.n_ieq_constr = 2

        super().__init__(vars=vars,
                         n_obj=1,
                         n_ieq_constr=self.n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):

        f = norm(self.obs_signal - self.dictionary @ self.var_vector)
        out["F"] = f

        if self.constraints:
            pos_constr = [self.t_max >= norm(self.pos_matrix @ self.var_vector) >= self.t_clip]
            neg_constr = [-self.t_max <= norm(self.neg_matrix @ self.var_vector) <= -self.t_clip]
            out["G"] = [pos_constr, neg_constr]


def solve_problem(obs_signal, dictionary, n_var, t_clip, t_max,
                  pos_matrix=None, neg_matrix=None, constraints=False):

    problem = SingleObjectiveOptimizationProblem(obs_signal, dictionary, n_var,
                                                 t_clip, t_max, pos_matrix, neg_matrix, constraints)
    sbx = SBX(prob=0.9, eta=15)
    algorithm = NSGA2(pop_size=40, n_offsprings=10, sampling=FloatRandomSampling(), crossover=sbx,
                      mutation=PM(eta=20), eliminate_duplicates=True)
    termination = get_termination("n_gen", 40)
    res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=False)
    x = res.X

    return x


def rectangular_window_function(length):

    window = ones((1, length))
    normalized = preprocessing.normalize(window, norm='l2')

    return normalized


def generate_dic(frame_len, n_atoms, window_function):

    window = window_function(frame_len)
    time = array(range(frame_len))
    frequency = array(range(n_atoms))

    dictionary = diag(window) * cos(pi / n_atoms * time.T * frequency)
    normalized = norm(dictionary)
    dictionary = divide(dictionary, normalized)

    return dictionary


def constrained_omp(obs_signal, m, dictionary, k_max, t_eps, t_clip, t_max):

    w = diag([norm(m @ dictionary)])
    dictionary_ = array(m @ dictionary @ w)
    k = 0
    sigma = []
    r = [obs_signal]

    while k != k_max or norm(r[k]) ** 2 >= t_eps:
        k += 1
        j = argmax(array([abs(dot(r[k - 1], dictionary_[j])) for j in range(len(dictionary_))]))
        sigma.append(j)
        x = solve_problem(obs_signal, dictionary_[sigma], len(sigma), t_clip, t_max)
        r.append(obs_signal - dictionary_[sigma] * x)

    pos_matrix = m_pos @ dictionary @ w
    neg_matrix = m_neg @ dictionary @ w
    x = solve_problem(obs_signal, dictionary_[sigma], len(sigma), t_clip, t_max,
                      pos_matrix, neg_matrix, constraints=True)

    return w * x


def plot_signal(full_path, csv_format, columns, title, start=0, end=-1):

    data = read_csv(full_path, sep=';')
    if csv_format == 'names':
        time = data[columns[0]]
        temp = data[columns[1]]
    else:
        time = data.iloc[start:end, columns[0]]
        temp = data.iloc[start:end, columns[1]]
    figure(title)
    plot(time, temp)


if __name__ == '__main__':

    plot_temp = True

    if plot_temp:

        base_path = 'C://Users//samue//Documents//ENSTA_Bretagne//Projet_Clipping//Signaux'

        numerical_file = f'{base_path}//Numerical//signal_T_R=0_therm_f2_DH36.txt'
        plot_signal(numerical_file, 'names', ['Time (s)', ' Temp_moy_ROI (K)'], 'Numerical')

        experimental_file = f'{base_path}//Exp//F15 5 9_540_fa100_f4_R=inc.txt'
        plot_signal(experimental_file, 'no_names', [1, 2], 'Experimental', 0, 500)

    show()
