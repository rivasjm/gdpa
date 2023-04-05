from analysis import HolisticAnalyis
from assignment import PassthroughAssignment
from examples import get_palencia_system
from gradient_descent import GDPA, invslack, Adam


def main():
    """
    Code for the simple running example in the paper, composed of 6 steps
    The idea is to run this in debug mode, step by step, and capture the numbers
    """

    # use Palencia example, but with 3 priority levels instead of 2
    system = get_palencia_system()
    priorities = [1, 2, 3, 1, 2, 1]
    for prio, task in zip(priorities, system.tasks):
        task.priority = prio

    # we also set the deadlines as D=T
    deadlines = [35, 45]
    for deadline, flow in zip(deadlines, system.flows):
        flow.deadline = deadline

    analysis = HolisticAnalyis(reset=False, limit_factor=10)
    lr = 3
    delta = 1.5
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.1
    gamma = 0.9

    gdpa = GDPA(verbose=True, initial=PassthroughAssignment(normalize=True),
                over_iterations=0,
                vectorized=False, # non vectorized to easily fetch the cost function values (with vectorized I get the same gradient)
                iterations=2, cost_fn=invslack, analysis=analysis, delta=delta,
                optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, gamma=gamma, seed=1))

    system.apply(gdpa)
    schedulable = system.is_schedulable()
    if schedulable and gdpa.iterations == 2:
        print(f"iterations={gdpa.iterations}, schedulable = {schedulable}")


if __name__ == '__main__':
    main()
