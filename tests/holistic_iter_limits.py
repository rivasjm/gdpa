from analysis import HolisticAnalyis
from examples import get_medium_system, get_big_system


def test(system):
    holistic = HolisticAnalyis(limit_factor=10)
    holistic.apply(system)


if __name__ == '__main__':
    # system = get_medium_system(utilization=0.99)
    system = get_big_system(utilization=0.8)
    test(system)