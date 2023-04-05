import unittest
from vholistic import jitter_matrix, successor_matrix, get_vectors, VectorHolisticAnalysis
import numpy as np
import examples
from assignment import PDAssignment
from analysis import HolisticAnalyis
from timeit import default_timer as timer
from datetime import timedelta
from random import Random


class VHolisticTest(unittest.TestCase):
    def test_jitter_matrix(self):
        successors = np.array([2, 3, -1, 5, 6, -1], dtype=np.int32).reshape((-1, 1))
        s = successor_matrix(successors)
        R = np.array([[5, 12, 25, 5, 10, 15],
                      [6, 13, 26, 6, 11, 16]]).transpose().ravel('F').reshape(2, 6, 1)
        J = jitter_matrix(s, R)
        self.assertEqual(J[0, 0, 0], 0)
        self.assertEqual(J[0, 2, 0], 12)
        self.assertEqual(J[0, 5, 0], 10)
        self.assertEqual(J[1, 0, 0], 0)
        self.assertEqual(J[1, 2, 0], 13)
        self.assertEqual(J[1, 5, 0], 11)

    def test_validity(self):
        """Test different priority assignments with the medium system"""
        random = Random(42)
        pd = PDAssignment(normalize=True)
        system = examples.get_medium_system(random=random, utilization=0.5)
        pd.apply(system)

        tasks = system.tasks
        n = len(tasks)
        priorities = get_vectors(system)[-1]
        p = np.tile(priorities, (1, n*2))

        delta = 0.3
        for i, task in enumerate(tasks):
            p[i, i*2] += delta
            p[i, i*2+1] -= delta

        #
        # use vector analysis for all scenarios
        #

        t1 = timer()
        vholistic = VectorHolisticAnalysis(limit_factor=10)
        vholistic.set_priority_scenarios(p)
        vholistic.apply(system)
        rvector = vholistic.full_response_times
        t2 = timer()
        print(f"Vectorized Holistic = {t2 - t1}")

        #
        # use classic holistic analysis for all scenarios
        #

        rhol = np.zeros_like(rvector)
        t1 = timer()
        holistic = HolisticAnalyis(limit_factor=10)
        all_p = np.hstack((priorities, p))  # all priorities: original from system + variations
        for i, col in enumerate(all_p.T):
            # assign these priorites to the system
            for task, prio in zip(system.tasks, col.ravel()):
                task.priority = prio
            # analyze
            holistic.apply(system)

            # save response times to wcrt matrix
            rhol[:, i] = np.array([task.wcrt for task in system.tasks])

        t2 = timer()
        print(f"Classic Holistic = {t2 - t1}")
        cond = np.allclose(rvector, rhol, 0.0001, 0.0001)
        if not cond:
            print(cond)

        self.assertTrue(cond)

    def test_anomaly_system(self):
        system = examples.generate_anomaly_system()
        PDAssignment().apply(system)

        vholistic = VectorHolisticAnalysis(limit_factor=10)
        p = np.array([3, 2, 1, 3, 2, 2, 4, 3, 1, 3, 4, 3, 1, 2, 4, 2, 4, 1, 1, 4]).reshape(-1, 1)
        vholistic.set_priority_scenarios(p)
        vholistic.apply(system)
        rv = vholistic.full_response_times[:, 1]

        print(rv)


if __name__ == '__main__':
    unittest.main()
