from model import System
from gurobipy import *
import numpy as np
from gradient_descent import invslack
from assignment import save_assignment, restore_assignment
from analysis import reset_wcrt
from exec_time import ExecTime


def apply_priority_matrix(model, pm):
    """
    Apply a priority matrix to a model: specific values of fixed priorities are assigned
    in each task so that the priority matrix is fulfilled.
    """
    tasks = model.tasks
    for task in tasks:
        task.priority = 1
    a, _ = pm.shape
    for i in range(a):
        for j in range(a):
            if pm[i][j]:
                tasks[i].priority += 1


def objective_function(model, priority_matrix, analysis):
    """
    Example of an objective function that we can apply in a system.
    In this case, it calculates the worst inverse slack of the system (invslack function) for a given priority matrix.
    To calculate the inverse slack, it is necessary to calculate worst-case response times. To do this,
    we use an analysis called Holistic.
    """
    # asigna prioridades de acuerdo a la matriz de prioridades
    apply_priority_matrix(model, priority_matrix)

    # borramos los tiempos de respuesta de peor caso (si los hubiera)
    reset_wcrt(model)

    # calculamos tiempos de respuesta de peor caso en todas las tareas (atributo wcrt de las tareas)
    analysis.apply(model)

    # calculamos métrica del sistema con los tiempos de respuesta de peor caso y los plazos
    metric = invslack(model)
    return metric


class GurobiBFAssignment:
    def __init__(self, analysis, verbose=False):
        self.analysis = analysis
        self.verbose = verbose
        self.system = None
        self.exec_time = ExecTime()

    def apply(self, system: System):
        self.exec_time.init()
        # clear system pointer
        self.system = None

        # launch optimization. this saves into system the best solution found
        self.calculate_candidates(system)

        # restore the best assignment found
        restore_assignment(system)
        self.exec_time.stop()

    def mycallback(self, model, where):
        if not self.system:
            return

        tasks = range(len(self.system.tasks))
        if where == GRB.Callback.MIPSOL:
            solution = model.cbGetSolution(model.getVars())
            varList = model.getAttr("varName", model.getVars())

            sol = {}
            for v in range(len(varList)):
                sol[varList[v]] = round(solution[v])

            temp = [(i, j) for i in tasks for j in tasks if
                    round(sol.get('Prio_[' + str(i) + ',' + str(j) + ']', 0), 0) == 1]

            finalP = np.zeros((len(self.system.tasks), len(self.system.tasks)), dtype=bool)
            for i in temp:
                finalP[i[0], i[1]] = True

            # perform the heuristic and quality check. It's stored in variable `quality'
            quality = objective_function(self.system, finalP, self.analysis)

            # You can also check here if the solution is good enough.
            # I don't know how you measure quality, so I leave this out.
            # If you are happy with the quality you get, you can use model.terminate() to terminate.
            if self.verbose:
                print(f"Evaluating possible solution: quality = {quality}")
            if quality < 0:
                save_assignment(self.system)
                if self.verbose:
                    print("Feasible solution found!")
                model.terminate()

    def calculate_candidates(self, system: System):
        # saves system so it is available in the callback
        self.system = system

        # prepare input data for gurobi
        alloc = list(map(lambda t: system.processors.index(t.processor), system.tasks))  # aka o
        tasks = range(len(system.tasks))                                                 # aka I
        cores = range(len(system.processors))

        # prepare gurobi variables
        env = Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        m = Model(env=env)
        p = m.addVars(tasks, tasks, vtype=GRB.BINARY, name="Prio_")  # Priority array (Nrows=Ncolums=Number of tasks)
        nt = [len(proc.tasks) for proc in system.processors]

        # gurobi constraints

        # If tasks i and j do not belong to the same core, P=0
        m.addConstrs((p[i, j] == 0 for i in tasks for j in tasks if alloc[i] != alloc[j]), "cPrio=0")
        m.addConstrs((p[i, i] == 0 for i in tasks), "cPrio=0_")

        # Number of True on each core (4 tasks->6 True, 5 tasks->10 True)
        m.addConstrs(
            (quicksum(p[i, j] for i in tasks for j in tasks if alloc[i] == c and alloc[j] == c and i != j) == (nt[c] * (nt[c] - 1)) / 2 for c in cores), "cPrio=1")

        # For the antisymmetric and transitive properties of the priority order relation
        m.addConstrs((p[i, j] + p[j, i] == 1 for i in tasks for j in tasks if alloc[i] == alloc[j] and i != j), "cPriob")
        m.addConstrs(
            (p[i, j] + p[j, k] - 1 <= p[i, k] for i in tasks for j in tasks for k in tasks if alloc[i] == alloc[j] == alloc[k] and i != j != k), "cPrioc")

        # for now do not use objective function
        # m.setObjective( quicksum(P[i,j] for i in I for j in I), GRB.MINIMIZE )

        m.setParam('PoolSearchMode', 2)
        m.setParam('PoolSolutions', 100000000)
        # max_time = 5000  # seconds
        # m.setParam('TimeLimit', max_time)

        # launch!
        # as soon a solutionis found, the callback finishes the optimization automatically
        # the callback already saves the best solution found in the given system
        m.optimize(lambda model, where: self.mycallback(model, where))
