import unittest
from examples import get_small_system
from random import Random
from assignment import PDAssignment
from vector.bf_assignment import BruteForceAssignment, brute_force_space_size
from milp.gurobi import GurobiBFAssignment
from analysis import HolisticAnalyis
import generator
from gradient_descent import invslack
from examples import get_palencia_system


class GurobiBFTest(unittest.TestCase):

    def test_gurobi(self):
        rnd = Random(1)
        analysis = HolisticAnalyis(reset=False, limit_factor=1)
        ga = GurobiBFAssignment(analysis=analysis, verbose=True)
        system = get_palencia_system()
        ga.apply(system)
        self.assertTrue(system.is_schedulable())

    def test_vs_brute_force(self):
        rnd = Random(1)
        analysis = HolisticAnalyis(reset=False, limit_factor=1)

        pd = PDAssignment()
        bf = BruteForceAssignment()
        ga = GurobiBFAssignment(analysis=analysis, verbose=False)

        for i in range(100):
            system = generator.generate_system(random=rnd, n_flows=4, n_tasks=4, n_procs=4,
                                               utilization=0.5,
                                               period_min=10, period_max=1000,
                                               deadline_factor_min=1, deadline_factor_max=1,
                                               balanced=True)

            pd.apply(system)
            analysis.apply(system)
            slack_pd = invslack(system)

            # print(f"Brute Force Space Size: {brute_force_space_size(system)}")
            bf.apply(system)
            analysis.apply(system)
            slack_bf = invslack(system)

            ga.apply(system)
            analysis.apply(system)
            slack_ga = invslack(system)

            print(f"{i}: pd={slack_pd:.2f} bf={slack_bf:.2f} ga={slack_ga:.2f} ")
            self.assertEqual(slack_bf < 0, slack_ga < 0)
