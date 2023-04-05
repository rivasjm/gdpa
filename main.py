from gradient_descent import GDPA, invslack, Adam
from assignment import PDAssignment, HOPAssignment
from analysis import HolisticAnalyis, reset_wcrt
from examples import get_small_system
from random import Random

# Simple example to try GDPA

# Generate a synthetic system
system = get_small_system(Random(4), utilization=0.8, balanced=True)

# Instantiate a Holistic analysis object as the schedulability test
sched_test = HolisticAnalyis(limit_factor=10)

# Instantatiate a HOPA object
hopa = HOPAssignment(analysis=HolisticAnalyis(reset=False, limit_factor=10), normalize=True)

# Instantiate a GDPA object
# initial priority assignment = PD
gdpa = GDPA(verbose=False, initial=PDAssignment(normalize=True),
            iterations=100, cost_fn=invslack, analysis=HolisticAnalyis(reset=False, limit_factor=10),
            delta=1.5, optimizer=Adam(lr=3, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=1.2))

# Apply HOPA on the generated system.
hopa.apply(system)
sched_test.apply(system)  # calculate worst-case response times
print(f"HOPA:  schedulable={system.is_schedulable()}, invslack={invslack(system)}")

# Apply GDPA on the generated system.
gdpa.apply(system)
reset_wcrt(system)  # delete previous worst-case response times
sched_test.apply(system)
print(f"GDPA:  schedulable={system.is_schedulable()}, invslack={invslack(system)}")
