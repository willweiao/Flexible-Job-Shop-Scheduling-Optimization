from src.fjsp_makespan import FJSPSolver
from src.plot import Plotter


def main():
    case_name = "mk01"

    solver = FJSPSolver(case_name)
    solver.solve()

    plotter = Plotter(case_name)
    plotter.plot_all()

if __name__ == "__main__":
    main()

