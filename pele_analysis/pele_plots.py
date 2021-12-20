import matplotlib.pyplot as plt
import numpy as np

def bindingLandscape(report_data, colormap='Blues_r', color_column='Ligand SASA', dpi=100, **kwargs):
    """
    Creates a binding energy landscape according to the relevant distance. A column
    can be used as color mapping.

    Parameters
    ==========
    report_data : pandas.DataFrame
        Pandas data frame returned by the readReportFiles() function.
    color_column : str
        Column to be mapped in a color dimmension
    dpi : int
        Plot resolution
    """
    fig, ax = plt.subplots(dpi=dpi)
    sp = report_data.reset_index().plot(kind='scatter',
                     x='Relevant Distance',
                     y='Binding Energy',
                     c=color_column,
                     colormap=colormap,
                     ax=ax,
                     **kwargs)
    plt.show()

def energyLandscape(report_data, colormap='Blues_r', color_column='Binding Energy', dpi=100,
                    x_delta_lim=50.0, ascending=False, **kwargs):
    """
    Creates a total energy landscape according to the relevant distance. A column
    can be used as color mapping.

    Parameters
    ==========
    report_data : pandas.DataFrame
        Pandas data frame returned by the readReportFiles() function.
    color_column : str
        Column to be mapped in a color dimmension
    dpi : int
        Plot resolution
    x_delta_lim : float
        Range of total energy to be displayed in the plot.
    ascending : bool
        Whether color column values must be sorted ascendingly (or descendingly)
        before plotting them?
    """

    fig, ax = plt.subplots(dpi=dpi)
    report_data.sort_values(color_column, ascending=ascending).plot(
                     kind='scatter',
                     c=color_column,
                     y='Total Energy',
                     x='Relevant Distance',
                     colormap=colormap,
                     ax=ax,
                     **kwargs)
    plt.ylim(int(report_data['Total Energy'].min()-(x_delta_lim*0.1)),
             int(report_data['Total Energy'].min()+(x_delta_lim*0.9)))
    plt.show()

def plotValuesByEpoch(report_data, column, title=None):
    """
    Plot distribution of values by epoch based on a report_data column. It draws
    only the lowest outlier for each distribution.

    Parameters
    ==========
    report_data : pandas.DataFrame
        Pandas data frame returned by the readReportFiles() function.
    column : str
        Column inside report_data containing the values to plot.
    title : str
        Plot title
    """
    column_values = []
    epochs = report_data.index.levels[0]
    for epoch in epochs:
        epoch_series = report_data[report_data.index.get_level_values('Epoch') == epoch]
        column_values.append(epoch_series[column])

    bp = plt.boxplot(column_values, labels=report_data.index.levels[0])
    plt.title(title)

    # Get only the lowest value outlier in the plot
    for fly, cv in zip(bp['fliers'], column_values):
        fdata = fly.get_data()
        average = np.average(cv)
        if fdata[1] != []:
            index = fdata[0][np.argmin(fdata[1])]
            f_value = np.min(fdata[1])

            if f_value < average:
                fdata = [[index], [f_value]]
                fly.set_data(fdata)
            else:
                fly.set_data([],[])

    ax = plt.gca()
    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()

    plt.xlabel('Epoch')
    plt.ylabel(column)
    plt.show()
