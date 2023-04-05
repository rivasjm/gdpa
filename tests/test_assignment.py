import unittest
from random import Random

import numpy as np
from examples import *
from assignment import PDAssignment, HOPAssignment, walk_random_priorities, walk_random_priorities_processors
from analysis import HolisticAnalyis
from generator import create_series, generate_system
from model import *


class PDAssignmentTest(unittest.TestCase):

    def test_palencia(self):
        system = get_palencia_system()
        flow1 = system['flow1']
        flow2 = system['flow2']

        #
        system.apply(PDAssignment())

        # test priorities in each processor
        self.assertGreater(flow1['a1'].priority, flow2['a6'].priority)
        self.assertGreater(flow1['a2'].priority, flow2['a5'].priority)
        self.assertLess(flow1['a3'].priority, flow2['a4'].priority)

        # for task in flow1:
        #     print(f"name={task.name}, proc={task.processor.name}, wcet={task.wcet}, priority={task.priority}")
        #
        # for task in flow2:
        #     print(f"name={task.name}, proc={task.processor.name}, wcet={task.wcet}, priority={task.priority}")


class HOPAAssignmentTest(unittest.TestCase):

    def test_palencia(self):
        system = get_palencia_system()
        analysis = HolisticAnalyis()
        hopa = HOPAssignment(analysis=analysis, verbose=True)
        system.apply(hopa)
        self.assertTrue(system.is_schedulable())

    def test_palencia_series(self):
        template = get_palencia_system()
        analysis = HolisticAnalyis()
        hopa = HOPAssignment(analysis=analysis, verbose=True)

        series = create_series(template, np.linspace(0.1, 0.99, 100))
        for system in series:
            print(f"Test: HOPA for Palencia with utilization={system.utilization}")
            system.apply(hopa)

    def test_random(self):
        random = Random(222)
        analysis = HolisticAnalyis()
        hopa = HOPAssignment(analysis=analysis, verbose=True)

        for _ in range(1, 5):
            n_flows = random.randint(1, 10)
            n_procs = random.randint(1, 5)
            n_tasks = random.randint(1, 10)
            utilization = random.uniform(0.1, 1.0)
            period_min = 100
            period_max = period_min * random.uniform(2.0, 1000.0)
            deadline_factor_min = 0.5
            deadline_factor_max = 2

            template = generate_system(random, n_flows, n_tasks, n_procs, utilization,
                                       period_min, period_max,
                                       deadline_factor_min, deadline_factor_max)
            series = create_series(template=template, utilizations=np.linspace(0.1, 0.9, num=50))

            for system in series:
                print(f"Test: HOPA for Random with utilization={system.utilization}")
                system.apply(hopa)


class MiscTests(unittest.TestCase):

    def test_walk_random_priorities(self):
        system = get_palencia_system()
        count = 0

        def inc(ignored_system):
            nonlocal count
            count += 1

        walk_random_priorities(system, 10, 10, inc)
        self.assertEqual(100, count)

    def test_walk_random_priorities_processors(self):
        system = get_barely_schedulable()
        count = 0

        def inc(ignored_system):
            nonlocal count
            count += 1

        analysis = HolisticAnalyis(limit_factor=10, reset=False)
        pd = PDAssignment()
        system.apply(pd)
        system.apply(analysis)
        initial_slack = system.slack

        walk_random_priorities_processors(system, 10, 10, inc)
        self.assertEqual(100, count)

        # confirm that the priorities and processor mappings are restored
        system.apply(analysis)
        self.assertEqual(initial_slack, system.slack)


if __name__ == '__main__':
    unittest.main()
