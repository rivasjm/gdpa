from model import System, Flow, Task, Processor
from examples import get_palencia_system
import generator
import model


def export(system: System, file):
    txt = write_system(system)
    with open(file, "w") as f:
        f.write(txt)


def write_system(system: System) -> str:
    header = """Model (
        Model_Name  => {0},
        Model_Date  => 2000-01-01);""".format("System")

    procs = "\n\n".join(map(write_processing_resource, system.processors))
    sss = "\n\n".join(map(write_scheduling_server, system.tasks))
    os = "\n\n".join(map(write_operation, system.tasks))
    ts = "\n\n".join(map(write_transaction, system.flows))
    return header + "\n\n" + procs + "\n\n" + sss + "\n\n" + os + "\n\n" + ts


def write_processing_resource(processor: Processor) -> str:
    return (
        f'Processing_Resource (\n'
        f'      Type 			            => Fixed_Priority_Processor,\n'
        f'      Name 			            => {processor.name},\n'
        f'      Max_Priority		        => 500,\n'
        f'      Min_Priority		        =>  1,\n'
        f'      Max_Interrupt_Priority	    => 512,\n'
        f'      Min_Interrupt_Priority	    => 501);')


def write_scheduling_server(task: Task) -> str:
    return """Scheduling_Server (
        Type				=> Fixed_Priority,
        Name 				=> {0},
        Server_Sched_Parameters		=> (
                Type		=> Fixed_Priority_policy,
                The_Priority	=> {1},
                Preassigned		=> no),
        Server_Processing_Resource	=> {2});""".format(task.name, task.priority, task.processor.name)


def write_operation(task: Task) -> str:
    return """Operation (
        Type        => Simple,
        Name        => {0},
        Worst_Case_Execution_Time   => {1});""".format(task.name, task.wcet)


def write_transaction(flow: Flow) -> str:
    prefix = """Transaction (
        Type	=> Regular, 
        Name	=> {0},
        External_Events => (
            (Type 		=> Periodic,
            Name 		=> {1},
            Period 	    => {2})),""".format(flow.name, external_event_name(flow), flow.period)

    ies = """
        Internal_Events => ({0}),""".format("".join(map(write_internal_event, flow.tasks)))

    ehs = """
        Event_Handlers => ({0})""".format(",".join(map(write_event_handler, flow.tasks)))

    return prefix + ies + ehs + ");"


def write_internal_event(task: Task) -> str:
    fixed = """
            (Type 	=> regular,
            name 	=> {0}""".format(output_event_name(task))
    if not task.is_last:
        return fixed + "),"
    else:
        return fixed + """,
            Timing_Requirements => (
                Type 		  => Hard_Global_Deadline,
                Deadline 	  => {0},
                referenced_event => {1}))""".format(task.flow.deadline, external_event_name(task.flow))


def write_event_handler(task: Task) -> str:
    return """
            (Type         => Activity,
            Input_Event         => {0},
            Output_Event        => {1},
            Activity_Operation  => {2},
            Activity_Server     => {3})""".format(input_event_name(task), output_event_name(task), task.name, task.name)


def output_event_name(task: Task) -> str:
    return f"o_{task.name}"


def reverse_output_event_name(event_name) -> str:
    return event_name.lstrip("o_")


def input_event_name(task: Task) -> str:
    return external_event_name(task.flow) if len(task.predecessors) == 0 else output_event_name(task.predecessors[0])


def external_event_name(flow: Flow) -> str:
    return f"e_{flow.name}"


def sanitize_priorities(system: System):
    """
    In-place sanitization of the priorities
    In MAST priorities must be integers higher than 0
    :param system:
    """
    # first save current priorities
    tasks = system.tasks
    model.save_attrs(tasks, ["priority"], "_sanitize_")

    # assign integer priorities in the same ordering
    prio = 1
    for task in sorted(tasks, key=lambda t: t.priority):
        task.priority = prio
        prio += 1


def desanitize_priorities(system: System):
    """
    Restore the un-sanitized priorities of the system
    :param system:
    """
    model.restore_attrs(system.tasks, ["priority"], "_sanitize_")


if __name__ == '__main__':
    system = get_palencia_system()
    export(system, "test.txt")
