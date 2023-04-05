from analysis import reset_wcrt, HolisticAnalyis
from assignment import PDAssignment, HOPAssignment, RandomAssignment
from gradient_descent import *
from generator import set_utilization
import itertools
import numpy as np
from examples import get_medium_system, get_big_system, get_small_system, get_system
from random import Random
from multiprocessing import Pool
from datetime import datetime
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from mast.mast_wrapper import MastOffsetAnalysis, MastHolisticAnalysis, MastOffsetPrecedenceAnalysis
import vector.bf_assignment
from evaluation import brute_force_anomaly


# size = (2, 10, 5)  # flows, tasks/flow, processors
# size = (2, 4, 2)  # flows, tasks/flow, processors
# size = (3, 5, 3)  # flows, tasks/flow, processors

# size = (30, 3, 11)  # flows, tasks/flow, processors
#
# lrs = [3]
# deltas = [1.5]
# beta1s = [0.9]
# beta2s = [0.999]
# epsilons = [0.1]
# population = 50
# utilization_min = 0.5
# utilization_max = 0.9
# utilization_steps = 20
#
# start = vector_times.vector_times()  # starting vector_times (seconds since epoch)


def achieves_schedulability(system, assignment, test) -> bool:
    reset_wcrt(system)
    system.apply(assignment)
    system.apply(test)
    return system.is_schedulable()


class GDPAComparison:
    def __init__(self, label, size, assignments, sched_test, population=50,
                 utilization_min=0.5, utilization_max=0.9, utilization_steps=20,
                 deadline_factor_min=0.5, deadline_factor_max=1,
                 threads=4):

        self.label = label
        self.size = size
        self.assignments = assignments
        self.sched_test = sched_test

        self.population = population
        self.utilization_min = utilization_min
        self.utilization_max = utilization_max
        self.utilization_steps = utilization_steps

        self.deadline_factor_min = deadline_factor_min
        self.deadline_factor_max = deadline_factor_max
        self.threads = threads

        self.start = time.time()

    def run(self):
        random = Random(42)
        utilizations = np.linspace(self.utilization_min, self.utilization_max, self.utilization_steps)
        systems = [get_system(self.size, random, balanced=True, name=str(i),
                              deadline_factor_min=self.deadline_factor_min,
                              deadline_factor_max=self.deadline_factor_max) for i in range(self.population)]
        names, _ = zip(*self.get_assignments())
        results = np.zeros((len(names), len(utilizations)))
        iterations = np.zeros((len(names), len(utilizations)))
        times = np.zeros((len(names), len(utilizations)))

        job = 0
        for u, utilization in enumerate(utilizations):
            # set utilization to the generated systems
            for system in systems:
                set_utilization(system, utilization)

            # launch jobs into the thread pool
            with Pool(self.threads) as pool:
                func = partial(self.step, index=u)
                for arr, iters, exec_time, index, system_name in pool.imap_unordered(func, systems):
                    job += 1
                    results[:, index] += arr
                    iterations[:, index] += iters
                    times[:, index] += exec_time
                    print(".", end="")

                    self.save_log(self.name, u, utilization, system_name, names, arr)
                    if job % 25 == 0:
                        print(f"\n{datetime.now()} : job={job}")
                    if job % self.population == 0:
                        self.save_files(results, f"{self.name}_scheds", names, utilizations)
                        self.save_files(np.divide(iterations, results, where=results != 0), f"{self.name}_iterations",
                                        names, utilizations, ylabel="Iterations to Schedule", show=False)
                        self.save_files(np.divide(times, results, where=results != 0), f"{self.name}_times",
                                        names, utilizations, ylabel="Execution times", show=False)
                        # excel(label, names, utilizations, results)
                        # chart(label, names, utilizations, results, save=True)

        # excel(label, names, utilizations, results)
        # chart(label, names, utilizations, results, save=True)
        self.save_files(results, f"{self.name}_scheds", names, utilizations)
        self.save_files(np.divide(iterations, results, where=results != 0), f"{self.name}_iterations",
                        names, utilizations, ylabel="Iterations to Schedule", show=False)
        self.save_files(np.divide(times, results, where=results != 0), f"{self.name}_times",
                        names, utilizations, ylabel="Execution times", show=False)

    def save_files(self, values, label, names, utilizations, ylabel="Schedulable systems", show=True):
        self.excel(label, names, utilizations, values)
        self.chart(label, names, utilizations, values, ylabel=ylabel, save=True, show=show)

    def save_log(self, label, u, utilization, system_name, tools, results):
        res_str = " ".join([t for t, r in zip(tools, results) if r > 0])
        with open(f"{label}_log.txt", "a") as f:
            line = f"{datetime.now()} : {label} {utilization:.2f}({u}) {system_name} \t-> {res_str}\n"
            f.write(line)
            # if brute_force_anomaly.has_anomaly(line):
            #     print("\nANOMALY: " + line, end="")

    def print_overview(label, names, utilizations, results):
        df = pd.DataFrame(data=results,
                          index=names,
                          columns=utilizations)
        print(df)

    def chart(self, label, names, utilizations, results, ylabel="Schedulable systems", save=False, show=True):
        plt.clf()
        # the export version should be transposed, it is the convention to have the continuous data in the columns
        df = pd.DataFrame(data=np.transpose(results),
                          index=utilizations,
                          columns=names)
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Average utilization")

        # print system size
        system_size = "size=" + "x".join(map(str, self.size))
        ax.annotate(system_size, xy=(0, -0.1), xycoords='axes fraction', ha='left', va="center", fontsize=8)

        # register execution vector_times
        time_label = f"{time.time() - self.start:.2f} seconds"
        ax.annotate(time_label, xy=(1, -0.1), xycoords='axes fraction', ha='right', va="center", fontsize=8)
        fig.tight_layout()
        if save:
            fig.savefig(f"{label}.png")
        if show:
            plt.show()

    def excel(self, label, names, utilizations, results):
        # the export version should be transposed, it is the convention to have the continuous data in the columns
        df = pd.DataFrame(data=np.transpose(results),
                          index=utilizations,
                          columns=names)
        df.to_excel(f"{label}.xlsx")

    def step(self, system, index):
        names, assigs = zip(*self.get_assignments())
        sched_test = self.get_sched_test()
        results = np.zeros(len(assigs), dtype=np.int)
        iterations = np.zeros(len(assigs))
        times = np.zeros(len(assigs))

        for a, assig in enumerate(assigs):
            sched = achieves_schedulability(system, assig, sched_test)
            if sched and assig.exec_time and assig.exec_time.has_time():
                times[a] += assig.exec_time.exec_time

            if sched:
                results[a] += 1

            if sched and hasattr(assig, "iterations_to_sched") and assig.iterations_to_sched > -1:
                iterations[a] += assig.iterations_to_sched

        return results, iterations, times, index, system.name

    def get_assignments(self):
        return self.assignments

    def get_sched_test(self):
        return self.sched_test

    @property
    def name(self):
        s = "".join(map(str, self.size))
        return f"{s}-{self.label}"
