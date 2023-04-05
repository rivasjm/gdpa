import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

extension = "pdf"


def add_text(ax, posx, posy, label, size='small', align='center'):
    ax.text(posx, posy, label, fontsize=size, horizontalalignment=align, transform=ax.transAxes,
            fontweight='bold', bbox=dict(boxstyle="round", ec='black', fc='bisque'))


def subfigure_labels(axs):
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    for i, a in enumerate(axs):
        a.text(-0.05, -0.1, labels[i], fontweight='bold', fontsize='medium', horizontalalignment='right',
               transform=a.transAxes)


def plot_vectorized_analysis():
    # load data
    medium = pd.read_excel("./vector_times/vector-time-medium.xlsx", index_col=0)  # 4x5=20 steps
    big = pd.read_excel("./vector_times/vector-time-big.xlsx", index_col=0)        # 8x8=64 steps

    medium['holistic-mast'] = medium['holistic-mast'] / medium['holistic']
    medium['holistic-vector'] = medium['holistic-vector'] / medium['holistic']
    medium['holistic'] = medium['holistic'] / medium['holistic']

    big['holistic-mast'] = big['holistic-mast'] / big['holistic']
    big['holistic-vector'] = big['holistic-vector'] / big['holistic']
    big['holistic'] = big['holistic'] / big['holistic']

    # prepare chart
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(3.5, 5))
    styles = ['+-', 'o-', '.--', 's:']

    # plot
    medium.drop(['holistic'], axis=1).plot.line(ax=axes[0], logy=True, logx=True, style=styles)
    big.drop(['holistic'], axis=1).plot.line(ax=axes[1], logy=True, logx=True, style=styles)

    # configure common properties of axes
    for i, ax in enumerate(axes):
        ax.set_ylabel("Normalized Execution Time", fontweight='bold', size='9')
        ax.set_xlabel("Number of Priority Assignments", fontweight='bold')
        ax.grid(True, which='major', axis='x')
        ax.legend(prop={'weight': 'bold', 'size': 9}, loc="center right")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    # particular axes properties
    axes[1].set_ylim([0.005, 1])
    add_text(axes[0], 0.8, 0.65, "20 steps", size='medium')
    add_text(axes[1], 0.8, 0.85, "64 steps", size='medium')
    subfigure_labels(axes)

    # save fig
    fig.savefig("vector-times." + extension)


def plot_gdpa_schedulables():
    # load data
    small = pd.read_excel("./comparison/444-small_scheds.xlsx", index_col=0)
    medium = pd.read_excel("./comparison/655-medium_scheds.xlsx", index_col=0)
    big = pd.read_excel("./comparison/1267-big_scheds.xlsx", index_col=0)

    # reorder columns of small, so the first 3 ones are same in the 3 scenarios
    small = small[['pd', 'hopa', 'gdpa-hopa', 'gdpa-pd', 'gdpa-random', 'brute-force']]
    medium = medium[['pd', 'hopa', 'gdpa-hopa', 'gdpa-pd', 'gdpa-random']]
    big = big[['pd', 'hopa', 'gdpa-hopa']]

    # rename "brute-force" column to "brute-force | milp"
    small.rename(columns={"brute-force": "brute-force | milp"}, inplace=True)

    # prepare chart
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(12, 3))
    styles = ['+-', 'o-', 'x--', 's:', '*-', 'o-']

    # plot
    small.plot.line(ax=axes[0], style=styles)
    medium.plot.line(ax=axes[1], style=styles)
    big.plot.line(ax=axes[2], style=styles)

    # configure common properties of axes
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_ylabel("Schedulable Systems", fontweight='bold')
        ax.set_xlabel("Average Utilization", fontweight='bold')
        ax.grid(True, which='major', axis='x')
        ax.legend(prop={'weight': 'bold', 'size': 9})

    # particular axes properties
    add_text(axes[0], 0.2, 0.6, "16 steps", size='medium')
    add_text(axes[1], 0.2, 0.45, "30 steps", size='medium')
    add_text(axes[2], 0.2, 0.35, "72 steps", size='medium')
    subfigure_labels(axes)

    # save fig
    fig.savefig("gdpa-comparison-scheds." + extension)


def plot_gdpa_times():
    # load data
    small = pd.read_excel("./comparison/444-small_times.xlsx", index_col=0)
    small_gurobi = pd.read_excel("./comparison/444-small-gurobi_times.xlsx", index_col=0)
    medium = pd.read_excel("./comparison/655-medium_times.xlsx", index_col=0)
    big = pd.read_excel("./comparison/1267-big_times.xlsx", index_col=0)

    # add gurobi-milp column to "small" dataframe
    small = small.join(small_gurobi["milp"])

    # reorder columns of small, so the first 3 ones are same in the 3 scenarios
    small = small[['pd', 'hopa', 'gdpa-hopa', 'gdpa-pd', 'gdpa-random', 'brute-force', 'milp']]
    medium = medium[['pd', 'hopa', 'gdpa-hopa', 'gdpa-pd', 'gdpa-random']]
    big = big[['pd', 'hopa', 'gdpa-hopa']]

    # prepare chart
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(12, 3))
    styles = ['+-', 'o-', 'x--', 's:', '*-', 'o-', '+-']

    # plot
    small.plot.line(ax=axes[0], style=styles, logy=True, legend=False)
    medium.plot.line(ax=axes[1], style=styles, logy=True, legend=False)
    big.plot.line(ax=axes[2], style=styles, logy=True, legend=False)

    # configure common properties of axes
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_ylabel("Avg. Computation Time (s)", fontweight='bold')
        ax.set_xlabel("Average Utilization", fontweight='bold')
        ax.set_ylim([1e-6, 1e6])
        ax.grid(True, which='major', axis='both')
        if i == 0:
            ax.legend(ncol=3, prop={'size': 8},
                      labelspacing=0.1, columnspacing=0.3, framealpha=1)
        else:
            ax.legend(ncol=3, prop={'size': 8, 'weight': 'bold'},
                      labelspacing=0.1, columnspacing=0.3, framealpha=1)

    # setup common legend
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=3, bbox_to_anchor=(0.665, 0.99), prop={'weight': 'bold', 'size': 7},
               # labelspacing=0.1, columnspacing=0.5)

    # particular axes properties
    add_text(axes[0], 0.15, 0.05, "16 steps", size='medium')
    add_text(axes[1], 0.15, 0.05, "30 steps", size='medium')
    add_text(axes[2], 0.15, 0.05, "72 steps", size='medium')
    subfigure_labels(axes)

    # save fig
    fig.savefig("gdpa-comparison-times." + extension)


def plot_gdpa_iterations():
    # load data
    small = pd.read_excel("./comparison/444-small_iterations.xlsx", index_col=0).drop(columns=['pd'])
    medium = pd.read_excel("./comparison/655-medium_iterations.xlsx", index_col=0).drop(columns=['pd'])
    big = pd.read_excel("./comparison/1267-big_iterations.xlsx", index_col=0).drop(columns=['pd'])

    # reorder columns of small, so the first 3 ones are same in the 3 scenarios
    small = small[['hopa', 'gdpa-hopa', 'gdpa-pd', 'gdpa-random']]
    medium = medium[['hopa', 'gdpa-hopa', 'gdpa-pd', 'gdpa-random']]
    big = big[['hopa', 'gdpa-hopa']]

    # prepare chart
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(12, 3))
    styles = ['o-', 'x--', 's:', '*-', 'o-']

    # get the colors from the active cycler
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # plot. remove first color, that is used in other plots for PD
    small.plot.line(ax=axes[0], style=styles, color=colors[1:])
    medium.plot.line(ax=axes[1], style=styles, color=colors[1:])
    big.plot.line(ax=axes[2], style=styles, color=colors[1:])

    # configure common properties of axes
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_ylabel("Avg. Iterations", fontweight='bold')
        ax.set_xlabel("Average Utilization", fontweight='bold')
        ax.grid(True, which='major', axis='both')
        ax.legend(prop={'weight': 'bold', 'size': 9}, loc='upper left')

    # particular axes properties
    add_text(axes[0], 0.2, 0.55, "16 steps", size='medium')
    add_text(axes[1], 0.2, 0.55, "30 steps", size='medium')
    add_text(axes[2], 0.2, 0.7, "72 steps", size='medium')
    subfigure_labels(axes)

    # save fig
    fig.savefig("gdpa-comparison-iterations." + extension)


def plot_gdpa_offsets():
    # load data
    medium = pd.read_excel("./comparison/655-medium-offsets_scheds.xlsx", index_col=0)

    # prepare chart
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 3))
    styles = ['+-', 'o-', 'x--', 's:', '*-', 'o-']

    # plot
    medium.plot.line(ax=ax, style=styles)

    # configure common properties of axes
    ax.set_ylabel("Schedulable Systems", fontweight='bold')
    ax.set_xlabel("Average Utilization", fontweight='bold')
    ax.grid(True, which='major', axis='x')
    ax.legend(prop={'weight': 'bold', 'size': 9})

    # particular axes properties
    add_text(ax, 0.2, 0.35, "30 steps", size='medium')

    # save fig
    fig.savefig("gdpa-comparison-scheds-offsets." + extension)


def plot_gdpa_evaluation():
    # Schedulables
    plot_gdpa_schedulables()
    
    # Running times
    plot_gdpa_times()

    # Number of iterations
    plot_gdpa_iterations()

    # GDPA with offset analysis
    plot_gdpa_offsets()


def main():
    #############################
    # VECTORIZED ANALYSIS TIMES #
    #############################
    plot_vectorized_analysis()

    ###################
    # GDPA EVALUATION #
    ###################
    plot_gdpa_evaluation()


if __name__ == '__main__':
    main()
