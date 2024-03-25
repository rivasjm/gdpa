import itertools

from paper.comparison.comparison_worker import GDPAComparison
from vector import bf_assignment
from analysis import HolisticAnalyis
from assignment import PDAssignment, HOPAssignment, RandomAssignment
from gradient_descent import GDPA, invslack, Adam, LRUpdater, NoiseUpdater
from mast.mast_wrapper import MastOffsetAnalysis
from milp.gurobi import GurobiBFAssignment


def gdpa_params():
    lrs = [3]
    deltas = [1.5]
    beta1s = [0.9]
    beta2s = [0.999]
    epsilons = [0.1]
    gammas = [1.2]
    return lrs, deltas, beta1s, beta2s, epsilons, gammas


def gdpa_params_fine():
    lrs = [3]
    deltas = [1.5]
    beta1s = [0.9]
    beta2s = [0.999]
    epsilons = [0.1]
    gammas = [0.9, 1.5, 5]
    return lrs, deltas, beta1s, beta2s, epsilons, gammas


def gdpa_params_study():
    lrs = [0.1, 3, 5]
    deltas = [1, 1.5, 2]
    beta1s = [0.7, 0.9]
    beta2s = [0.7, 0.999]
    epsilons = [0.01, 0.1, 0.2]
    gammas = [0.9, 1.2, 1.5, 3]
    return lrs, deltas, beta1s, beta2s, epsilons, gammas


def get_full_gdpa_name(prefix, lr, delta, beta1=None, beta2=None, epsilon=None, gamma=None):
    ret = f"{prefix}-lr{lr}-d{delta}"
    if beta1:
        ret += f"-b1{beta1}"
    if beta2:
        ret += f"-b2{beta2}"
    if epsilon:
        ret += f"-e{epsilon}"
    if gamma:
        ret += f"-g{gamma}"

    return ret


def get_assignments(analysis, pd=False, hopa=False, bf=False,
                    gdpa_r=False, gdpa_pd=False, gdpa_hopa=False,
                    gurobi=False,
                    gdpa_adam=False, gdpa_noise=False, gdpa_direct=False,
                    full_gdpa_names=False,
                    params_getter=gdpa_params):
    assigs = []
    # legacy assignments
    pd_assig = PDAssignment(normalize=True)
    if pd:
        assigs.append(("pd", pd_assig))

    hopa_assig = HOPAssignment(analysis=analysis, normalize=True)
    if hopa:
        assigs.append(("hopa", hopa_assig))

    # brute force assignments
    if bf:
        brute = bf_assignment.BruteForceAssignment(size=10000)
        assigs.append(("brute-force", brute))

    # gurobi milp
    if gurobi:
        gurobi = GurobiBFAssignment(analysis=HolisticAnalyis(limit_factor=1, reset=False))
        assigs.append(("milp", gurobi))

    # GDPA assignments
    lrs, deltas, beta1s, beta2s, epsilons, gammas = params_getter()

    params = itertools.product(lrs, deltas, beta1s, beta2s, epsilons, gammas)
    for lr, delta, beta1, beta2, epsilon, gamma in params:
        # GDPA Random
        if gdpa_r:
            assig = GDPA(verbose=False, initial=RandomAssignment(normalize=True),
                         iterations=100, cost_fn=invslack, analysis=analysis, delta=delta,
                         optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, gamma=gamma))
            prefix = "gdpa-random"
            label = get_full_gdpa_name(prefix, lr, delta, beta1, beta2, epsilon, gamma) if full_gdpa_names else prefix
            assigs.append((label, assig))

        # GDPA PD
        if gdpa_pd:
            assig = GDPA(verbose=False, initial=pd_assig,
                         iterations=100, cost_fn=invslack, analysis=analysis, delta=delta,
                         optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, gamma=gamma))
            prefix = "gdpa-pd"
            label = get_full_gdpa_name(prefix, lr, delta, beta1, beta2, epsilon, gamma) if full_gdpa_names else prefix
            assigs.append((label, assig))

        # GDPA HOPA
        if gdpa_hopa:
            assig = GDPA(verbose=False, initial=hopa_assig,
                         iterations=100, cost_fn=invslack, analysis=analysis, delta=delta,
                         optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, gamma=gamma))
            prefix = "gdpa-hopa"
            label = get_full_gdpa_name(prefix, lr, delta, beta1, beta2, epsilon, gamma) if full_gdpa_names else prefix
            assigs.append((label, assig))

    # Special case of GDPA: Adam without noise
    if gdpa_adam:
        for lr, delta, beta1, beta2, epsilon in itertools.product(lrs, deltas, beta1s, beta2s, epsilons):
            assig = GDPA(verbose=False, initial=pd_assig,
                         iterations=100, cost_fn=invslack, analysis=analysis, delta=delta,
                         optimizer=Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, noise=False))
            prefix = "gdpa-adam"
            label = get_full_gdpa_name(prefix, lr, delta, beta1, beta2, epsilon) if full_gdpa_names else prefix
            assigs.append((label, assig))

    # Special case of GDPA: Only Noise
    if gdpa_noise:
        for lr, delta, gamma in itertools.product(lrs, deltas, gammas):
            assig = GDPA(verbose=False, initial=pd_assig,
                         iterations=100, cost_fn=invslack, analysis=analysis, delta=delta,
                         optimizer=NoiseUpdater(lr=lr, gamma=gamma))
            prefix = "gdpa-noise"
            label = get_full_gdpa_name(prefix, lr, delta, gamma=gamma) if full_gdpa_names else prefix
            assigs.append((label, assig))

    # Special case of GDPA: Not Adam, Not Noise
    if gdpa_direct:
        for lr, delta in itertools.product(lrs, deltas):
            assig = GDPA(verbose=False, initial=pd_assig,
                         iterations=100, cost_fn=invslack, analysis=analysis, delta=delta,
                         optimizer=LRUpdater(lr=lr))
            prefix = "gdpa-direct"
            label = get_full_gdpa_name(prefix, lr, delta) if full_gdpa_names else prefix
            assigs.append((label, assig))

    return assigs


def do_gdpa_parameters_small():
    # size: flows, steps/flow, procs
    size = (4, 4, 4)  # 16 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, hopa=True, gdpa_pd=True, params_getter=gdpa_params_study, full_gdpa_names=True)

    # perform comparison
    worker = GDPAComparison("parameters-small", size, assigs, sched_test, save_figs=False, threads=4)
    worker.run()


def do_gdpa_optimizers_small():
    # size: flows, steps/flow, procs
    size = (4, 4, 4)  # 16 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, hopa=True, gdpa_pd=True, gdpa_adam=True, gdpa_noise=True, gdpa_direct=True)

    # perform comparison
    worker = GDPAComparison("optimizers-small", size, assigs, sched_test)
    worker.run()


def do_gdpa_optimizers_medium():
    # size: flows, steps/flow, procs
    size = (6, 5, 5)  # 30 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, hopa=True, gdpa_pd=True, gdpa_adam=True, gdpa_noise=True, gdpa_direct=True,
                             full_gdpa_names=True, params_getter=gdpa_params_fine)

    # perform comparison
    worker = GDPAComparison("optimizers-medium", size, assigs, sched_test)
    worker.run()


def do_small_gurobi():
    # size: flows, steps/flow, procs
    size = (4, 4, 4)  # 16 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, gurobi=True)

    # perform comparison
    worker = GDPAComparison("small-gurobi", size, assigs, sched_test, threads=1)
    worker.run()


def do_small():
    # size: flows, steps/flow, procs
    size = (4, 4, 4)  # 16 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, pd=True, hopa=True, bf=True,
                             gdpa_r=True, gdpa_pd=True, gdpa_hopa=True)

    # perform comparison
    worker = GDPAComparison("small", size, assigs, sched_test)
    worker.run()


def do_medium():
    # size: flows, steps/flow, procs
    size = (6, 5, 5)  # 30 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, pd=True, hopa=True, gdpa_r=True, gdpa_pd=True, gdpa_hopa=True)

    # perform comparison
    worker = GDPAComparison("medium", size, assigs, sched_test, threads=1)
    worker.run()


def do_big():
    # size: flows, steps/flow, procs
    size = (12, 6, 7)  # 72 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, pd=True, hopa=True, gdpa_hopa=True)

    # perform comparison
    worker = GDPAComparison("big", size, assigs, sched_test)
    worker.run()


def do_comparison_offsets():
    # size: flows, steps/flow, procs
    size = (6, 5, 5)  # 30 steps

    # schedulability test to evaluate the final solution
    sched_test = MastOffsetAnalysis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, pd=True, hopa=True, gdpa_hopa=True)

    # perform comparison
    worker = GDPAComparison("medium-offsets", size, assigs, sched_test)
    worker.run()


def do_big_industrial():
    # size: flows, steps/flow, procs
    size = (35, 3, 12)  # 72 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, pd=True, hopa=True, gdpa_hopa=True)

    # perform comparison
    worker = GDPAComparison("big-industrial", size, assigs, sched_test,
                            utilization_min=0.3, utilization_max=0.7,
                            deadline_factor_min=1/size[1], deadline_factor_max=1/size[1])  # D=T
    worker.run()


def do_small_industrial():
    # size: flows, steps/flow, procs
    size = (8, 3, 8)  # 72 steps

    # schedulability test to evaluate the final solution
    sched_test = HolisticAnalyis(limit_factor=1)

    # analysis used to internally evaluate iterations
    analysis = HolisticAnalyis(reset=False, limit_factor=10)

    # store assignments in a list of name,tool pairs
    assigs = get_assignments(analysis, pd=True, hopa=True, gdpa_hopa=True, bf=True)

    # perform comparison
    worker = GDPAComparison("big-industrial", size, assigs, sched_test,
                            utilization_min=0.3, utilization_max=0.7,
                            deadline_factor_min=1/size[1], deadline_factor_max=1/size[1])  # D=T
    worker.run()


if __name__ == '__main__':
    ##################
    # GDPA OPTIMIZER #
    ##################

    # do_gdpa_optimizers_small()
    # do_gdpa_optimizers_medium()
    do_gdpa_parameters_small()

    ######################
    # GENERAL COMPARISON #
    ######################

    # do_small()
    # do_small_gurobi()
    # do_medium()
    # do_big()  # this one takes a few days to complete

    ########################
    # SPECIFIC EVALUATIONS #
    ########################

    # do_comparison_offsets()    # medium size with offset analysis evaluating final solution
    # do_small_industrial()      # 24 steps, with short flows, d=t, w/ brute-force, mimics industrial examples (1679616)
    # do_big_industrial()        # many steps (>100) with short flows, may procs (>10), d=t, mimics industrial examples


