from random import Random

import examples
from analysis import reset_wcrt
from gradient_descent import invslack
from mast_wrapper import MastHolisticAnalysis, MastOffsetPrecedenceAnalysis, MastOffsetAnalysis
from assignment import walk_random_priorities, PDAssignment
from examples import get_medium_system
import matplotlib.pyplot as plt
import math


class CorrRegister:
    def __init__(self, analysis1, analysis2, name1, name2):
        self.analysis1 = analysis1
        self.analysis2 = analysis2
        self.name1 = name1
        self.name2 = name2
        self.results1 = []
        self.results2 = []

    def apply(self, system):
        reset_wcrt(system)
        self.analysis1.apply(system)
        r1 = invslack(system)

        reset_wcrt(system)
        self.analysis2.apply(system)
        r2 = invslack(system)

        if r1 < 1e90 and r2 < 1e90:
            self.results1.append(r1)
            self.results2.append(r2)
            self.plot()

    def plot(self, file=None):
        plt.scatter(self.results1, self.results2)
        plt.xlabel(self.name1)
        plt.ylabel(self.name2)
        plt.show()
        if file:
            plt.savefig(file)


def holistic_offsetpr_corr():
    """MAST Holistic vs MAST Offset PR correlation"""
    a1 = MastHolisticAnalysis(limit_factor=10)
    a2 = MastOffsetAnalysis(limit_factor=10)
    register = CorrRegister(a1, a2, "holistic", "offsets")

    system = get_medium_system(random=Random(1), utilization=0.8)
    pd = PDAssignment()
    system.apply(pd)

    walk_random_priorities(system, 20, 5, register, seed=42)
    register.plot("hol_offpr.png")


def holistic_mast_corr():
    """MAST Holistic vs Python Holistic correlation"""
    pass


if __name__ == '__main__':
    holistic_offsetpr_corr()