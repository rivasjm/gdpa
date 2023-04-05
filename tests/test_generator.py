import unittest
from generator import uunifast, set_processor_utilization, log_uniform, generate_system, copy, create_series, \
    walk_series
from random import Random
import numpy as np
from examples import get_palencia_system


class GeneratorTest(unittest.TestCase):
    def test_uunifast(self):
        random = Random(11)
        for n_tasks in range(1, 100):
            for u in np.linspace(0.1, 1.0, num=100):
                us = uunifast(random, n_tasks, u)
                self.assertEqual(len(us), n_tasks)
                self.assertAlmostEqual(sum(us), u, delta=0.00001)

    def test_set_processor_utilization(self):
        system = get_palencia_system()
        for proc in system.processors:
            for u in np.linspace(0.1, 1.0, num=100):
                set_processor_utilization(proc, u)
                self.assertAlmostEqual(proc.utilization, u, delta=0.00001)

    def test_log_uniform(self):
        random = Random(2)
        for i in range(1000):
            low = random.uniform(10, (i+1)*10)
            high = low*(i+2)
            value = log_uniform(random, low, high)
            self.assertGreater(value, low)
            self.assertLess(value, high)

    def test_generate_system(self):
        random = Random(42)

        for _ in range(1, 10000):
            n_flows = random.randint(1, 10)
            n_procs = random.randint(1, 5)
            n_tasks = random.randint(1, 10)
            utilization = random.uniform(0.1, 1.0)
            period_min = 100
            period_max = period_min * random.uniform(2.0, 1000.0)
            deadline_factor_min = 0.5
            deadline_factor_max = 2

            system = generate_system(random, n_flows, n_tasks, n_procs, utilization,
                                     period_min, period_max,
                                     deadline_factor_min, deadline_factor_max)

            self.assertAlmostEqual(system.utilization, utilization, delta=0.000001)
            self.assertEqual(len(system.flows), n_flows)
            # if some processor has no task, it does not show up in system
            self.assertLessEqual(len(system.processors), n_procs)

    def test_copy(self):
        random = Random(42)

        for _ in range(1, 10000):
            n_flows = random.randint(1, 10)
            n_procs = random.randint(1, 5)
            n_tasks = random.randint(1, 10)
            utilization = random.uniform(0.1, 1.0)
            period_min = 100
            period_max = period_min * random.uniform(2.0, 1000.0)
            deadline_factor_min = 0.5
            deadline_factor_max = 2

            original = generate_system(random, n_flows, n_tasks, n_procs, utilization,
                                       period_min, period_max,
                                       deadline_factor_min, deadline_factor_max)

            system = copy(original)
            self.assertAlmostEqual(system.utilization, utilization, delta=0.000001)
            self.assertEqual(len(system.flows), n_flows)
            # if some processor has no task, it does not show up in system
            self.assertLessEqual(len(system.processors), n_procs)

    def test_series(self):
        random = Random(222)

        for _ in range(1, 100):
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
            series = create_series(template=template, utilizations=np.linspace(0.1, 1.0, num=100))

            for system, u in zip(series, np.linspace(0.1, 1.0, num=100)):
                self.assertAlmostEqual(system.utilization, u, delta=0.00001)
                for proc in system.processors:
                    self.assertAlmostEqual(proc.utilization, u, delta=0.00001)

    def test_walk_series(self):
        random = Random(333)
        count = 0

        def inc(system):
            nonlocal count
            count += 1
            self.assertLessEqual(system.utilization, 1.01)
            self.assertGreaterEqual(system.utilization, 0.099)

        for _ in range(50):
            n_flows = random.randint(1, 15)
            n_procs = random.randint(1, 4)
            n_tasks = random.randint(1, 15)
            utilization = 0.5  # ignored in the series
            period_min = 100
            period_max = period_min * random.uniform(2.0, 1000.0)
            deadline_factor_min = 0.5
            deadline_factor_max = 2

            template = generate_system(random, n_flows, n_tasks, n_procs, utilization,
                                       period_min, period_max,
                                       deadline_factor_min, deadline_factor_max)
            walk_series(system=template, utilizations=np.linspace(0.1, 1.0, num=100), callback=inc)
            self.assertAlmostEqual(0.5, template.utilization, delta=0.0001)

        self.assertEqual(5000, count)


if __name__ == '__main__':
    unittest.main()
