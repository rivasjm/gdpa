from analysis import HolisticAnalyis
from assignment import HOPAssignment, PDAssignment
from examples import get_system
from random import Random

from gradient_descent import GDPA, Adam, invslack

if __name__ == '__main__':
    """
    Objective: keep utilization constant at 70% while addind tasks
    Start eith 20 tasks, end with 100 for example
    
    I see it is tricky to make HOPA fail at 70%, maybe I should simply
    use GDPA with random start to force GDPA iterations
    """

    sched_test = HolisticAnalyis(limit_factor=1)
    analysis = HolisticAnalyis(reset=False, limit_factor=10)
    pd = PDAssignment(normalize=True)
    hopa = HOPAssignment(analysis=analysis, normalize=True)

    lr = 3
    delta = 1.5
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.1
    gdpa_pd = GDPA(verbose=False, initial=pd,
                   iterations=100, cost_fn=invslack, analysis=analysis, delta=delta,
                   optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, seed=1))
    gdpa_hopa = GDPA(verbose=False, initial=hopa,
                     iterations=100, cost_fn=invslack, analysis=analysis, delta=delta,
                     optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, seed=1))

    tasks, procs = (10, 5)
    random = Random(42)
    for flows in range(1, 21):
        s = get_system((flows, tasks, procs), random, utilization=0.7, balanced=False)

        print(f"tasks={len(s.tasks)}:", end=" ")
        gdpa_hopa.apply(s)
        print("ASSIGNED", end=" ")
        sched_test.apply(s)
        if s.is_schedulable():
            print(f"SCHEDULABLE", end=" ")
        print(f"iterations={gdpa_hopa.iterations_to_sched} time={gdpa_hopa.exec_time.exec_time}")




