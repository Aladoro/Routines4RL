import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_data(data, value="mean", xaxis='Epoch', yaxis='Reward',
              title=None):
    plt.figure(figsize=(10, 7))
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1)
    palette = np.array(sns.color_palette("deep"))
    ax = sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition",
                    color=palette)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if title is not None:
        plt.title(title)
    plt.legend(bbox_to_anchor=(0.00, -0.10),
               loc='center left',
               ncol=6)

    plt.tight_layout()
    plt.show()


def get_datasets(fpath, condition=None, show_random=False, epochs=100, plot_value_priorities=[None],
                 plot_value_name='plot_value', smooth=1, aggregate_envs=False):
    unit = 0
    datasets = []
    for root, dir, files in sorted(os.walk(fpath)):
        if 'log.txt' in files:
            if condition is not None:
                exp_name = condition
            else:
                exp_name = os.path.dirname(root)
                if aggregate_envs:
                    bn = os.path.basename(exp_name)
                    if bn[0].isupper():  # Assume experiment folders start with upper case
                        exp_name = bn
                    else:
                        exp_name = os.path.basename(os.path.dirname(exp_name))
                    split_exp_name = exp_name.split('_')
                    len_spl = len(split_exp_name)
                    exp_name = '_'.join(split_exp_name[:min(len_spl, 4)])

            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)
            if not show_random:
                print('drop')
                experiment_data = experiment_data[1:]
            n_data_points = len(experiment_data.index)
            if epochs > 0:
                if n_data_points > epochs:
                    print("WARNING: The maximum number of timesteps specified is less than data length")
            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
            )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                np.tile(exp_name,
                        n_data_points)
            )
            if plot_value_priorities[0] is not None:
                set_value = False
                i = 0
                while not set_value:
                    if plot_value_priorities[i] in experiment_data:
                        experiment_data.insert(
                            len(experiment_data.columns),
                            plot_value_name,
                            experiment_data[plot_value_priorities[i]]
                        )
                        set_value = True
                    i += 1
            if smooth > 1:
                filter = np.ones(smooth)
                normalizer = np.ones(experiment_data.values.shape[0])
                smoothed_data = np.convolve(experiment_data[plot_value_name], filter, 'same') / np.convolve(normalizer,
                                                                                                            filter,
                                                                                                            'same')
                experiment_data[plot_value_name] = smoothed_data
            if epochs > 0:
                if n_data_points > epochs:
                    new_exp_data = pd.DataFrame(experiment_data.values[:epochs])
                else:
                    new_exp_data = pd.DataFrame(np.concatenate([experiment_data.values,
                                                                np.repeat(experiment_data.tail(1).values,
                                                                          epochs - n_data_points, axis=0)]))
            else:
                new_exp_data = experiment_data
                epochs = n_data_points

            new_exp_data.columns = experiment_data.columns
            new_exp_data['Iteration'] = np.arange(epochs)
            datasets.append(new_exp_data)
            unit += 1
    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot data from experiments logs.')
    parser.add_argument('logdir', nargs='*', help='Names of directories containing the experiment logs for '
                                                  'the different tested algorithms.')
    parser.add_argument('--legend', nargs='*', help='Names of the different tested algorithms for plot legend.')
    parser.add_argument('--value', default='episode_returns_mean', help='Value to plot.')

    parser.add_argument('--value_priorities', nargs='*', help='Ordered list of preferred values we might plot for '
                                                              'each experiment. Used if some experiments do not include'
                                                              ' our preferred value to plot and we want to fall back '
                                                              'on a secondary value.')
    parser.add_argument('--plot_values', nargs='*', help='List of values to plot for each experiment.')

    parser.add_argument('--xaxis', default='Epoch', help='X-axis label.')
    parser.add_argument('--yaxis', default='Reward', help='Y-axis label.')
    parser.add_argument('--title', default=None, help='Plot title.')
    parser.add_argument('--show_random', action='store_true', help='Show the performance at timestep 0 with randomly '
                                                                   'initialized agent.')
    parser.add_argument('--epochs', default=0, help='Number of epochs to plot.')

    parser.add_argument('--smooth', default=1, help='Filter size for smoothing the plot.')
    parser.add_argument('--aggregate_envs', action='store_true', help='Aggregate scores of each algorithm from '
                                                                      'different environments')

    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True
    if args.value_priorities is not None:
        assert args.plot_values is None, 'value_priorities incompatible with plotting different values per directory'
        plotted_value = 'plot_value'
        args.plot_values = [args.value_priorities for _ in range(len(args.logdir))]

    elif args.plot_values is None:
        args.plot_values = [[None] for _ in range(len(args.logdir))]
        plotted_value = args.value
    else:
        assert len(args.plot_values) == len(args.logdir), \
            "Must give a plot_value for each set of experiments."
        args.plot_values = [[value] for value in args.plot_values]
        plotted_value = 'plot_value'
    data = []
    if use_legend:
        for logdir, legend_title, plot_value_priorities in zip(args.logdir, args.legend, args.plot_values):
            data += get_datasets(logdir, legend_title,
                                 show_random=args.show_random,
                                 epochs=int(args.epochs),
                                 plot_value_priorities=plot_value_priorities,
                                 plot_value_name=plotted_value,
                                 smooth=int(args.smooth),
                                 aggregate_envs=args.aggregate_envs, )
    else:
        for logdir, plot_value_priorities in zip(args.logdir, args.plot_values):
            data += get_datasets(logdir,
                                 show_random=args.show_random,
                                 epochs=int(args.epochs),
                                 plot_value_priorities=plot_value_priorities,
                                 plot_value_name=plotted_value,
                                 smooth=int(args.smooth),
                                 aggregate_envs=args.aggregate_envs)

    if isinstance(plotted_value, list):
        values = plotted_value
    else:
        values = [plotted_value]
    for value in values:
        plot_data(data, value=value, xaxis=args.xaxis, yaxis=args.yaxis,
                  title=args.title)


if __name__ == "__main__":
    main()
