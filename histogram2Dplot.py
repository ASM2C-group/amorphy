import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import matplotlib_style
# import matplotlib.gridspec as gridspec
import matplotlib_style

def snsplot(x, y, xlabel=None, ylabel=None, xlim=None, ylim=None, title=None):
    import seaborn as sns
    
    xlim == None
    if xlim == None or ylim == None:
        kdeplot = sns.jointplot(x=x, y=y, kind='kde', xlim=(np.min(x), np.max(x)), ylim=(np.min(y), np.max(y)) ,
                                cbar=True,  cmap=plt.get_cmap("gist_ncar_r"),  n_levels=30, fill=True, hue_norm=colors.Normalize(vmin=0, vmax=1), 
                                marginal_kws=dict(color="white"))
    else:
        kdeplot = sns.jointplot(x=x, y=y, kind='kde', xlim=(xlim[0], xlim[1]), ylim=(ylim[0], ylim[1]) ,
                                cbar=True,  cmap=plt.get_cmap("gist_ncar_r"),  n_levels=30, fill=True, hue_norm=colors.Normalize(vmin=0, vmax=1), 
                                marginal_kws=dict(color="white"))


    kdeplot.plot_marginals(sns.histplot, kde=True, bins=50)
    
    if xlabel == None or ylabel == None:
        kdeplot.set_axis_labels('', '', fontsize=26)
    else:
        kdeplot.set_axis_labels(xlabel, ylabel, fontsize=26)

    
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = kdeplot.ax_joint.get_position()
    pos_marg_x_ax = kdeplot.ax_marg_x.get_position()

    # reposition the joint ax so it has the same width as the marginal x ax
    kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])

    # reposition the colorbar using new x positions and y positions of the joint ax
    kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])

    plt.show()


def matplotlib(x, y, bins=40):
    plt.rcParams['font.family']='Times New Roman'
    # plt.rcParams['font.size']=24
    # plt.rcParams['lines.linewidth']=2
    # plt.rcParams['lines.markersize']=2
    # #plt.rcParams['axes.grid']=True
    # plt.rcParams['axes.labelsize']=28
    # plt.rcParams['axes.linewidth']=2
    # plt.rcParams['xtick.labelsize']=24
    # plt.rcParams['ytick.labelsize']=24
    # plt.rcParams['axes.grid']=True
    # plt.rcParams['xtick.direction']='out'
    # plt.rcParams['ytick.direction']='out'
    # plt.rcParams['xtick.minor.visible']=True
    # plt.rcParams['ytick.minor.visible']=True
    # plt.rcParams['xtick.major.width']=2
    # plt.rcParams['ytick.major.width']=2
    # plt.rcParams['xtick.minor.width']=2
    # plt.rcParams['ytick.minor.width']=2
    # plt.rcParams['xtick.major.size']=10
    # plt.rcParams['ytick.major.size']=10
    # plt.rcParams['xtick.minor.size']=5
    # plt.rcParams['ytick.minor.size']=5
    # plt.rcParams['xtick.top']=True
    # plt.rcParams['ytick.right']=True
    # plt.rcParams['legend.loc']='upper left'
    # plt.rcParams['legend.fontsize']=24
    cmap = plt.get_cmap("gist_ncar_r")
    
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    
    h,x3,y3 = np.histogram2d(y, x, bins=bins)
    h = h/np.max(h)
    
    
    def matplotlib_histogram2d(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
    
        ax_histx.tick_params(axis="y", labelleft=False)
        ax_histy.tick_params(axis="x", labelbottom=False)
    
        ax_histx.axis("off")
        ax_histy.axis("off")
    
        # the imshow plot:
        image = ax.imshow(h, origin="lower", interpolation="gaussian", norm=colors.Normalize(vmin=0, vmax=1), aspect="auto", extent=[x_min,x_max,y_min,y_max],  cmap=cmap)
    
        fig.colorbar(image, ax=ax)
    
        ax.set_xlabel("O-W distance (Å)", fontsize=26)
        ax.set_ylabel("Energy", fontsize=26)
        ax.set_xticks(np.round(np.linspace(np.min(x), np.max(x), 10), 1))
    
    
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')
    
    
    # Start with a square Figure.
    fig = plt.figure(figsize=(12, 12))
    
        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
    
    
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    
        # Create the Axes.
    ax       = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    plt.close()
        # Draw the scatter plot and marginals.
    #matplotlib_histogram2d(x, y, ax, ax_histx, ax_histy)
    
    
        # Create a Figure, which doesn't have to be square.
    fig = plt.figure(constrained_layout=True)
        # Create the main axes, leaving 25% of the figure space at the top and on the
        # right to position marginals.
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        # The main axes' aspect can be fixed.
    ax.set(aspect=1)
    
        # Create marginal axes, which have 25% of the size of the main axes.  Note that
        # the inset axes are positioned *outside* (on the right and the top) of the
        # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
        # less than 0 would likewise specify positions on the left and the bottom of
        # the main axes.
    
    ax_histx = ax.inset_axes([0, 1.01, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.01, 0, 0.25, 1], sharey=ax)
        # Draw the scatter plot and marginals.
    matplotlib_histogram2d(x, y, ax, ax_histx, ax_histy)
    plt.show()
