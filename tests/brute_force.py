import vector.bf_assignment as bf
from assignment import HOPAssignment
from analysis import HolisticAnalyis, reset_wcrt
from examples import get_system
from random import Random
from generator import set_utilization
from paper.comparison.comparison_worker import achieves_schedulability
import numpy as np

def find_system_problem():
    random = Random(42)
    utilizations = np.linspace(0.5, 0.9, 20)
    analysis = HolisticAnalyis(limit_factor=1)
    brute = bf.BruteForceAssignment(size=10000, verbose=True)
    hopa = HOPAssignment(analysis=HolisticAnalyis(reset=False, limit_factor=10), normalize=False, verbose=False)

    size = (2, 10, 5)  # flows, tasks/flow, processors
    systems = [get_system(size, random, balanced=True) for _ in range(50)]

    for u in utilizations:
        for n, system in enumerate(systems):
            set_utilization(system, u)

            m = f"{system.flows[0]} {system.flows[1]}"
            print(m)
            p = " ".join(map(lambda p: str(len(p.tasks)), system.processors))

            # brute.apply(system)
            # print(f"{n}:\tbrute={brute.schedulable} ", end="")
            # analysis.apply(system)
            # print(f"analysis={system.is_schedulable()} -> ", end="")
            # print(priorities_repr(system), end="")
            # if brute.schedulable != system.is_schedulable():
            #     print("############# ERROR ##############")
            # print("")
            # hopa.apply(system)
            # analysis.apply(system)
            # print(f"\thopa={system.is_schedulable()} -> {priorities_repr(system)}")

            # brute_sched = achieves_schedulability(system, brute, analysis)
            # prios = priorities_repr(system)
            # hopa_sched = achieves_schedulability(system, hopa, analysis)
            # msg = "ERROR" if hopa_sched and not brute_sched else ""
            # print(f"{u}:{n}: brute={brute_sched} hopa={hopa_sched} {msg} -> {prios}")


def brute_force_problem():
    system = get_problem_system()
    analysis = HolisticAnalyis(limit_factor=1)
    brute = bf.BruteForceAssignment(size=10000, verbose=True)
    hopa = HOPAssignment(analysis=HolisticAnalyis(reset=False, limit_factor=10), normalize=True, verbose=True)
    brute_sched = achieves_schedulability(system, brute, analysis)
    hopa_sched = achieves_schedulability(system, hopa, analysis)
    if brute_sched != hopa_sched:
        print("problema confirmado")


def get_problem_system():
    random = Random(42)
    size = (2, 10, 5)  # flows, tasks/flow, processors
    systems = [get_system(size, random, balanced=True) for _ in range(50)]
    system = systems[44]
    set_utilization(system, 0.5)
    return system


def priorities_repr(system):
    return " ".join(map(lambda t: str(int(t.priority)), system.tasks))


if __name__ == '__main__':
    # print(get_problem_system())
    find_system_problem()
