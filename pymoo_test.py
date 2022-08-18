from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real
from pymoo.optimize import minimize

from matplotlib.pyplot import plot, show
from numpy import linspace


class TestProblem(ElementwiseProblem):

    def __init__(self, points):

        self.points = points
        self.n_var = 2

        real_number = Real(bounds=(-10, 10))
        arr = [_ for _ in range(self.n_var)]
        typ = [real_number for _ in range(self.n_var)]
        vars = dict(zip(arr, typ))

        super().__init__(vars=vars,
                         n_obj=1)

    def _evaluate(self, X, out, *args, **kwargs):

        a, b = [X[i] for i in range(self.n_var)]

        f = sum((a * point[0] + b - point[1]) ** 2 for point in self.points)
        out["F"] = f


def solve_problem(points, gen):

    problem = TestProblem(points)
    algorithm = MixedVariableGA(pop_size=100, n_offsprings=100,
                                crossover=SBX(prob=0.9, eta=15),
                                survival=RankAndCrowdingSurvival())
    termination = get_termination("n_gen", gen)
    res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=False)
    x = res.X
    f = res.F

    return x, f


if __name__ == '__main__':

    demo_optimization = True

    if demo_optimization:

        values = [[1, 1.4], [2, 2.3], [3, 3.4],
                  [4, 4.1], [5, 5.1], [6, 6.8]]

        x = linspace(0, 7, 2)

        colors = ['blue', 'green', 'red', 'orange']
        gens = [1, 2, 3, 40]
        for color, gen in zip(colors, gens):
            solution, minimum = solve_problem(values, gen)
            plot(x, solution[0] * x + solution[1], color)
            print(f'Optimum: {round(minimum[0], 2)} \t| reached with {gen} generations ({color})')

        for point in values:
            plot(point[0], point[1], markersize=10, marker='.', color='k')

    show()
