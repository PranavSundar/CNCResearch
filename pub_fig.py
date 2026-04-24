"""
Publication-quality figure formatting for matplotlib plots.
Applies consistent styling across all research plots.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def pub_fig(ax=None, fig=None):
    """
    Apply publication-quality formatting to matplotlib axes and figure.

    Parameters:
    ax : matplotlib.axes.Axes, optional
        The axes to format. If None, uses plt.gca()
    fig : matplotlib.figure.Figure, optional
        The figure to format. If None, uses plt.gcf()
    """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # Set axis tight and enable grid
    ax.axis('tight')
    ax.grid(True, which='both', color=[0.8, 0.8, 0.8], alpha=0.25)

    # Font settings
    font_props = {'family': 'sans-serif', 'weight': 'bold', 'size': 15}
    try:
        if 'Helvetica Neue' in [f.name for f in fm.fontManager.ttflist]:
            font_props['family'] = 'Helvetica Neue'
    except Exception:
        pass

    # Apply font settings to labels, title, and tick labels
    ax.set_xlabel(ax.get_xlabel(), fontdict=font_props)
    ax.set_ylabel(ax.get_ylabel(), fontdict=font_props)
    ax.set_title(ax.get_title(), fontdict=font_props)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
        label.set_fontweight('bold')
        if font_props['family'] == 'Helvetica Neue':
            label.set_fontname('Helvetica Neue')

    # Tick parameters and line widths
    ax.tick_params(axis='both', which='major', labelsize=15, width=3, length=6, direction='out', top=False, right=False)

    # Spine (axis line) thickness - set thinner than current, a bit thicker than grid lines
    for spine_key, spine in ax.spines.items():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # White background for figure and axes
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Ensure grid is drawn on top of plot elements
    ax.set_axisbelow(False)
    ax.set_axisbelow(False)
    ax.set_frame_on(True)

    # Attempt to center the figure window for interactive use
    try:
        manager = fig.canvas.manager
        if hasattr(manager, 'window'):
            window = manager.window
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()
            win_w = window.winfo_width()
            win_h = window.winfo_height()
            window.geometry(f'+{(screen_w - win_w) // 2}+{(screen_h - win_h) // 2}')
    except Exception:
        pass  # skip when not supported