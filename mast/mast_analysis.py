import subprocess
from enum import Enum
import os.path
from mast.mast_results import parse_results
import mast.mast_writer as mast_writer
import uuid
import examples

# ROOT = "E:/dev/pymast/"
ROOT = "D:/dev/pymast/"
TEMP = "mast/temp/"
MAST_PATH = "mast/mast-1-5-1-0-bin/"
# MAST_PATH = "D:/dev/pymast/mast/mast-1-5-1-0-bin/"
MAST_EXECUTABLE = "mast_analysis.exe"
MAST_NON_SCHEDULABLE = "Final analysis status: NOT-SCHEDULABLE"
MAST_SCHEDULABLE = "The system is schedulable"
LIMIT = 1e100


class MastAnalysis(Enum):
    HOLISTIC = "holistic"
    OFFSET = "offset_based_approx"
    OFFSET_PR = "offset_based_approx_w_pr"


class MastAssignment(Enum):
    NONE = None
    PD = "pd"
    HOSPA = "hospa"
    SA = "annealing"


def analyze(system, analysis: MastAnalysis, assignment: MastAssignment = MastAssignment.NONE, limit=None):
    # create random temporary file names for this analysis, will be removed afterwards
    name = str(uuid.uuid1())
    input = os.path.abspath(os.path.join(ROOT, TEMP, name + ".txt"))
    output = os.path.abspath(os.path.join(ROOT, TEMP, name + "-out.xml"))
    preserve = False

    try:
        # make sure priorities are correct for mast (integers higher than 0)
        mast_writer.sanitize_priorities(system)

        # export system to a file with mast format
        mast_writer.export(system, input)

        # analyze with mast, capture results
        schedulable, results = run(analysis, assignment, input, output, limit)

        # save wcrts into the system
        for task in system.tasks:
            task.wcrt = results[task.name] if task.name in results else LIMIT

        # sanity check: system schedulability must match
        if system.is_schedulable() != schedulable:
            print("assertion error: " + input)
            preserve = True

    finally:
        # clean-up process: restore original unsanitized priorities, remove temporary files
        mast_writer.desanitize_priorities(system)
        if not preserve:
            clear_files(input, output)


def clear_files(*files):
    for file in files:
        if os.path.isfile(file):
            while True:
                try:
                    os.remove(file)
                    break
                except PermissionError:
                    pass


def run(analysis, assignment, input, output=None, limit=None, timeout=None):
    cmd = [os.path.join(ROOT, MAST_PATH, MAST_EXECUTABLE), analysis.value]
    if assignment is not MastAssignment.NONE:
        cmd.append("-p")
        cmd.append("-t")
        cmd.append(assignment.value)
    if limit:
        cmd.append("-f")
        cmd.append(str(limit))
    cmd.append(input)
    if output:
        cmd.append(output)

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        print("Timeout!")

    out = proc.stdout.read().decode() if proc and proc.stdout else ""
    schedulable = MAST_SCHEDULABLE in out if out else False
    results = parse_results(output) if output and os.path.isfile(output) else {}
    return schedulable, results


if __name__ == '__main__':
    system = examples.get_medium_system()
    # sched = run(MastAnalysis.HOLISTIC, MastAssignment.NONE, "test.txt", "test-out2.xml")
    # print(sched)
    analyze(system, MastAnalysis.HOLISTIC, MastAssignment.HOSPA)
    print(system.is_schedulable())