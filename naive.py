
from sympy.geometry import Line, intersection
import matplotlib.pyplot as plt
from numpy import array, sqrt, divide
from sklearn import preprocessing
from numpy.linalg import norm


def plot_line(line, color):
    """Plots a line consisting of a list a points, using matplotlib."""
    xs, ys = [], []
    for point in line:
        xs.append(point[0])
        ys.append(point[1])
        plt.plot(xs, ys, color=color)


def declipping(signal):
    """Returns a geometrically declipped signal."""
    declipped = []
    m = max([abs(point[1]) for point in signal])
    i = 0
    while i < len(signal) - 1:
        current, next = signal[i], signal[i + 1]
        if abs(current[1]) == m and abs(next[1]) == m:
            previous, after_next = signal[i - 1], signal[i + 2]
            line1, line2 = Line(previous, current), Line(next, after_next)
            vertices = intersection(line1, line2)[0]
            declipped.append((vertices.x, vertices.y))
            i += 2
        else:
            declipped.append(current)
            i += 1
    declipped.append(signal[-1])

    return declipped


if __name__ == '__main__':
    # sound = [[0, 0], [1, 0], [2, 4], [3, 4], [4, -4], [5, -4], [6, -1],
    #          [6.5, -3], [7, -1], [8, -4], [9, -1], [10, -2], [11, 1], [12, 0],
    #          [13, 2], [14, -3], [15, 4], [17, 4], [19, -3], [20, 0], [21, 0]]
    # answer = declipping(sound)
    #
    # plot_line(answer, 'green')
    # plot_line(sound, 'red')
    # plt.show()
    arr1 = array([1, 2, 3])
    arr2 = norm(arr1)
    print(arr2)
    ans = arr1 / arr2
    print(ans)
