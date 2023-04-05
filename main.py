from gradient_descent import GDPA, invslack, Adam
from assignment import PDAssignment, HOPAssignment
from analysis import HolisticAnalyis, reset_wcrt
from examples import get_medium_system
from random import Random

# Generate system
system = get_medium_system(Random(10), utilization=0.9, balanced=True)

# Instantiate a Holistic analysis object as the schedulability test
sched_test = HolisticAnalyis(limit_factor=1)

# Instantatiate a HOPA object
hopa = HOPAssignment(analysis=HolisticAnalyis(reset=False, limit_factor=10), normalize=True)

# Instantiate a GDPA object
gdpa = GDPA(verbose=False, initial=PDAssignment(normalize=True),
            iterations=100, cost_fn=invslack, analysis=HolisticAnalyis(reset=False, limit_factor=10),
            delta=1.5, optimizer=Adam(lr=3, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=1.2))

# Apply HOPA on the generated system.
# It tries to find a fixed-priority mapping for every task such that the system becomes schedulable
hopa.apply(system)

# Determine if HOPA was able to find a schedulable priority assignment
sched_test.apply(system)
print(f"HOPA is schedulable: {system.is_schedulable()}")

# Apply GDPA on the generated system.
# It tries to find a fixed-priority mapping for every task such that the system becomes schedulable
gdpa.apply(system)

# Determine if HOPA was able to find a schedulable priority assignment
reset_wcrt(system)
sched_test.apply(system)
print(f"GDPA is schedulable: {system.is_schedulable()}")