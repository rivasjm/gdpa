from analysis import HolisticAnalyis, reset_wcrt
from vector.vholistic import VectorHolisticAnalysis
from mast.mast_wrapper import MastHolisticAnalysis
import time
import numpy as np
from examples import get_medium_system, get_big_system
from assignment import PDAssignment
from random import Random
import pandas as pd
import matplotlib.pyplot as plt


def step(system, scenarios):
    """
    Measures the execution vector_times of analyzing a number of priority scenarios
    :param system:
    :param scenarios:
    :return: (mast vector_times, holistic vector_times, vector vector_times) tuple
    """
    mast = MastHolisticAnalysis(limit_factor=10)
    hol = HolisticAnalyis(limit_factor=10)
    vec = VectorHolisticAnalysis(limit_factor=10)

    # note: perf_counter returns vector_times in fractional seconds:
    # https://docs.python.org/3/library/time.html#time.perf_counter

    # MAST Holistic: sequentially analyze each priority scenario
    before = time.perf_counter()
    for _ in range(scenarios):
        system.apply(mast)
    mast_time = time.perf_counter() - before

    # Holistic: sequentially analyze each priority scenario
    before = time.perf_counter()
    for _ in range(scenarios):
        system.apply(hol)
    hol_time = time.perf_counter() - before

    # Vector Holistic: analyze all the scenarios in parallel
    before = time.perf_counter()
    t = len(system.tasks)
    if scenarios > 1:
        priorities = np.array([task.priority for task in system.tasks]).reshape((t, 1)).repeat(scenarios-1, axis=1)
        vec.set_priority_scenarios(priorities)
    system.apply(vec)
    vec_time = time.perf_counter() - before

    return np.asarray((mast_time, hol_time, vec_time))


def run(systems, name):
    print(f"Running {name}")
    pd = PDAssignment()
    scenarios = [1, 10, 100, 1000, 10000]
    tools = ["holistic-mast", "holistic", "holistic-vector"]
    results = np.zeros(shape=(len(tools), len(scenarios)), dtype=np.float64)

    for s, system in enumerate(systems):
        print(f"  system {s}:", end="")
        system.apply(pd)
        for i, scenario in enumerate(scenarios):
            print(f" {scenario}", end="")
            res = step(system, scenario)
            results[:, i] += res
            save(results, s+1, tools, scenarios, name)
        print()


def save(results, number, columns, index, name):
    df = pd.DataFrame(results.T, columns=columns, index=index) / number
    df.to_excel(f"{name}.xlsx")

    fig, ax = plt.subplots()
    df.plot(ax=ax, xlabel="number of scenarios", ylabel="execution vector_times (s)",
            use_index=True, logy=True, logx=True, figsize=(6, 3))
    fig.tight_layout()
    fig.savefig(f"{name}.png")
    fig.show()


if __name__ == '__main__':
    # medium-size systems
    random = Random(42)
    mediums = [get_medium_system(random, utilization=0.7) for _ in range(10)]
    run(mediums, "vector-vector_times-medium")

    # big-size systems
    random = Random(71)
    bigs = [get_big_system(random, utilization=0.7) for _ in range(10)]
    run(bigs, "vector-vector_times-big")
