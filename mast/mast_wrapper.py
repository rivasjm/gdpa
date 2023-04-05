from mast.mast_analysis import MastAnalysis, MastAssignment, analyze
from model import System


class MastWrapper:
    def __init__(self, analysis, assignment=MastAssignment.NONE, limit_factor=100):
        self.analysis = analysis
        self.assignment = assignment
        self.limit_factor = limit_factor

    def apply(self, system: System) -> None:
        analyze(system, self.analysis, self.assignment, limit=self.limit_factor)


class MastHolisticAnalysis(MastWrapper):
    def __init__(self, assignment=MastAssignment.NONE, limit_factor=100):
        super().__init__(MastAnalysis.HOLISTIC, assignment=assignment, limit_factor=limit_factor)


class MastOffsetAnalysis(MastWrapper):
    def __init__(self, assignment=MastAssignment.NONE, limit_factor=100):
        super().__init__(MastAnalysis.OFFSET, assignment=assignment, limit_factor=limit_factor)


class MastOffsetPrecedenceAnalysis(MastWrapper):
    def __init__(self, assignment=MastAssignment.NONE, limit_factor=100):
        super().__init__(MastAnalysis.OFFSET_PR, assignment=assignment, limit_factor=limit_factor)


if __name__ == '__main__':
    import examples
    import assignment

    system = examples.get_medium_system()
    pd = assignment.PDAssignment()
    # # analysis = MastHolisticAnalysis()
    # analysis = MastOffsetPrecedenceAnalysis()
    # hopa = assignment.HOPAssignment(analysis=analysis)
    # system.apply(hopa)
    # system.apply(analysis)
    # print(system.is_schedulable())

    sa = MastHolisticAnalysis(assignment=MastAssignment.SA)
    pd.apply(system)
    print(system.is_schedulable())
    sa.apply(system)
    print(system.is_schedulable())
