from random import Random
from generator import generate_system, set_utilization
from model import *
import numpy as np


def get_palencia_system() -> System:
    system = System()

    # 2 cpus + 1 network
    cpu1 = Processor(name="cpu1")
    cpu2 = Processor(name="cpu2")
    network = Processor(name="network")
    system.add_procs(cpu1, cpu2, network)

    # priority levels
    HIGH = 10
    LOW = 1

    # 2 flows
    flow1 = Flow(name="flow1", period=30, deadline=60)
    flow2 = Flow(name="flow2", period=40, deadline=80)

    # tasks
    flow1.add_tasks(
        Task(name="a1", wcet=5, priority=HIGH, processor=cpu1),
        Task(name="a2", wcet=2, priority=LOW, processor=network),
        Task(name="a3", wcet=20, priority=LOW, processor=cpu2)
    )
    flow2.add_tasks(
        Task(name="a4", wcet=5, priority=HIGH, processor=cpu2),
        Task(name="a5", wcet=10, priority=HIGH, processor=network),
        Task(name="a6", wcet=10, priority=LOW, processor=cpu1)
    )
    system.add_flows(flow1, flow2)
    system.name = "palencia"
    return system


def get_cruise_control():
    # From "Enabling Scheduling Analysis for AUTOSAR Systems", Anssi 2011
    system = System()

    # 2 cpus + 1 network
    body = Processor(name="body")
    engine = Processor(name="engine")
    can = Processor(name="can")
    system.add_procs(body, engine, can)

    # 2 flows
    flow1 = Flow(name="flow1", period=10, deadline=70)
    flow2 = Flow(name="flow2", period=10, deadline=30)

    # add tasks
    flow1.add_tasks(
        Task(name="acq", wcet=2.5, priority=1, processor=body),
        Task(name="inter", wcet=2.32, priority=1, processor=body),
        Task(name="message", wcet=1.52, priority=1, processor=can),
        Task(name="speed", wcet=1.5, priority=2, processor=engine),
        Task(name="cond", wcet=2, priority=3, processor=engine),
        Task(name="basic", wcet=1, priority=3, processor=engine),
        Task(name="contr", wcet=1, priority=3, processor=engine)
    )
    flow2.add_tasks(
        Task(name="diag", wcet=1.52, priority=4, processor=body),
        Task(name="message2", wcet=2, priority=4, processor=can),
        Task(name="limp", wcet=0.5, priority=3, processor=engine)
    )
    system.add_flows(flow1, flow2)
    system.name = "cruise-control"
    return system


def get_system(size, random=Random(), utilization=0.5, balanced=False, name=None,
               deadline_factor_min=0.5, deadline_factor_max=1) -> System:
    n_flows, t_tasks, n_procs = size
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=utilization,
                             period_min=100,
                             period_max=100*3,
                             deadline_factor_min=deadline_factor_min,
                             deadline_factor_max=deadline_factor_max,
                             balanced=balanced)
    system.name = name
    return system


def get_barely_schedulable() -> System:
    random = Random(123)
    n_flows, t_tasks, n_procs = (4, 5, 3)
    return get_system((n_flows, t_tasks, n_procs), random, 0.84, name="barely")


def get_small_system(random=Random(), utilization=0.5, balanced=False) -> System:
    n_flows, t_tasks, n_procs = (3, 4, 3)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="small")


def get_medium_system(random=Random(), utilization=0.84, balanced=False) -> System:
    n_flows, t_tasks, n_procs = (4, 5, 3)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="medium")


def get_big_system(random=Random(), utilization=0.84, balanced=False) -> System:
    n_flows, t_tasks, n_procs = (8, 8, 5)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="big")
