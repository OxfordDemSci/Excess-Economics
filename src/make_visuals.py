"""Make the figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def center_ylabels(ax1, ax2):
    """Centre labels between two axes."""
    pos2 = ax2.get_position()
    right = pos2.bounds[0]
    pos1 = ax1.get_position()
    left = pos1.bounds[0] + pos1.bounds[2]
    offset = ((right - left) / pos2.bounds[2]) * -0.45
    for yt in ax2.get_yticklabels():
        yt.set_position((offset, yt.get_position()[1]))
        yt.set_ha('center')
        plt.setp(ax2.yaxis.get_major_ticks(), pad=0)


def plot_func(age_forecasts, allreg_df, age_df,
              heatmap_df, fig_path, filename):
    """Make the figure."""
    fig = plt.figure(figsize=(14, 13))
    ax1 = plt.subplot2grid((32, 32), (0, 0), colspan=15, rowspan=7)
    ax2 = plt.subplot2grid((32, 32), (8, 0), colspan=15, rowspan=7)
    ax3 = plt.subplot2grid((32, 32), (0, 16), colspan=7, rowspan=15)
    ax4 = plt.subplot2grid((32, 32), (0, 24), colspan=7, rowspan=15)
    ax5 = plt.subplot2grid((32, 32), (18, 0), colspan=7, rowspan=15)
    ax6 = plt.subplot2grid((32, 32), (18, 8), colspan=7, rowspan=15)
    ax7 = plt.subplot2grid((32, 32), (18, 17), colspan=14, rowspan=15)
    color = ['#377eb8', '#ffb94e']

    # A.
    age_forecasts.index = age_forecasts.index.str.replace('201', "'1")
    age_forecasts.index = age_forecasts.index.str.replace('202', "'2")
    start_spot = 140
    annot_spot = 223-start_spot
    legend_elements = [Line2D([0], [0], color=color[1], lw=1,
                       label='Observed', alpha=1),
                       Patch(facecolor=color[0], edgecolor='k',
                             lw=0.5, label='90% CIs', alpha=0.2)]
    all = 'allage_empratio_allreg'
    age_forecasts[[all + '_m']][start_spot:].plot(ax=ax1, legend=False,
                                                  color=[color[1]])
    ax1.fill_between(age_forecasts[start_spot:].index,
                     age_forecasts[all + '_m_ci_right'][start_spot:],
                     age_forecasts[all + '_m_ci_left'][start_spot:],
                     alpha=0.2)
    ax1.set_xlabel('')
    ax1.axvline(x=annot_spot, linewidth=1, color='k', linestyle='--',
                alpha=.35, dashes=(12, 6))
    ax1.annotate("Viral outbreak", xy=(annot_spot, 63.65), xycoords='data',
                 xytext=(annot_spot-37, 63.6), fontsize=10, textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax1.legend(handles=legend_elements, loc='upper left', frameon=False)
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    age_forecasts[[all + '_w']][start_spot:].plot(ax=ax2, legend=False,
                                                  color=[color[1]])
    ax2.fill_between(age_forecasts[start_spot:].index,
                     age_forecasts[all + '_w_ci_right'][start_spot:],
                     age_forecasts[all + '_w_ci_left'][start_spot:], alpha=0.2)
    ax2.set_xlabel('')
    ax2.xaxis.set_tick_params(rotation=10)
    ax2.axvline(x=annot_spot, linewidth=1, color='k', linestyle='--',
                alpha=.35, dashes=(12, 6))
    ax2.annotate("Viral outbreak", xy=(annot_spot, 53.25), xycoords='data',
                 xytext=(annot_spot-37, 53.125), fontsize=10,
                 textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax1.set_ylabel('Emp-Pop Ratio: M')
    ax2.set_ylabel('Emp-Pop Ratio: F')
    ax2.legend(handles=legend_elements, loc='upper left', frameon=False)

    # B
    errors = age_df['ci_w'].to_frame('forc_w')
    age_df[['obs_w', 'forc_w']].plot(kind='barh', xerr=errors, ax=ax3,
                                     legend=False, alpha=0.8, color=color,
                                     edgecolor='k').invert_xaxis()
    ax3.set_xlabel('Employment Ratio: F')
    errors = age_df['ci_m'].to_frame('forc_m')
    age_df[['obs_m', 'forc_m']].plot(kind='barh', xerr=errors, ax=ax4,
                                     color=color, edgecolor='k', alpha=0.8)
    ax4.set_xlabel('Employment Ratio: M')
    ax3.set_yticks([])
    ax4.set_yticklabels([' 16-17', ' 18-24', ' 25-34', ' 35-49', ' 50-64',
                         '  65+'], fontsize=8)
    ax4.legend(frameon=True, edgecolor='k', framealpha=1, loc='lower right',
               labels=['Observed', 'Forecast'])

    # C
    allreg_df = allreg_df.sort_values(by='w_diff')
    ax5.errorbar(allreg_df['w_diff'], list(range(0, 11)),
                 xerr=allreg_df['w_ci'], fmt='.k', markersize=11, marker='o',
                 mfc='w', mec=color[0], mew=1, linewidth=1, capsize=5)
    ax5.set_yticklabels(allreg_df.index.to_list())
    ax5.set_yticks(np.arange(len(allreg_df)))
    allreg_df = allreg_df.sort_values(by='w_diff')
    ax6.errorbar(allreg_df['m_diff'], list(range(0, 11)),
                 xerr=allreg_df['m_ci'], fmt='.k', markersize=11, marker='o',
                 mfc='w', mec=color[1], mew=1, linewidth=1, capsize=5)
    ax6.set_yticklabels(allreg_df.index.to_list())
    ax6.set_yticks([])
    ax5.set_xlabel('Excess Economic Loss: F')
    ax6.set_xlabel('Excess Economic Loss: M')
    ax5.axvline(x=0, linewidth=0.75, color='k', linestyle='--', alpha=0.25,
                dashes=(12, 6))
    ax6.axvline(x=0, linewidth=0.75, color='k', linestyle='--', alpha=0.25,
                dashes=(12, 6))
    ax6.set_yticklabels([])

    # D.
    n_colors = 256
    palette = sns.diverging_palette(220, 20, n=n_colors)
    color_min, color_max = [-heatmap_df.max().max(), heatmap_df.max().max()]

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    corr = pd.melt(heatmap_df.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    x = corr['x']
    y = corr['y']
    size = corr['value'].abs()
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
    size_scale = 50
    ax7.scatter(x=x.map(x_to_num), y=y.map(y_to_num), s=size * size_scale,
                c=corr['value'].apply(value_to_color), marker='o')
    ax7.set_xticks([x_to_num[v] for v in x_labels])
    ax7.set_xticklabels(x_labels, rotation=0)
    ax7.set_xticklabels(['16-17', '18-24', '25-34', '35-49', '50-64', '65+'],
                        fontsize=9)
    ax7.set_yticks([y_to_num[v] for v in y_labels])
    ax7.set_yticklabels(y_labels)
    ax7.grid(False, 'minor', alpha=0, zorder=100)
    ax7.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax7.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    axins = inset_axes(ax7, width="5%", height="100%", loc='right',
                       bbox_to_anchor=(0.15, 0.05, .875, 0.95),
                       bbox_transform=ax7.transAxes, borderpad=0)
    col_x = [0]*len(palette)
    bar_y = np.linspace(color_min, color_max, n_colors)
    bar_height = bar_y[1] - bar_y[0]
    axins.barh(y=bar_y, width=[5]*len(palette), left=col_x, height=bar_height,
               color=palette, linewidth=0)
    axins.set_ylim(bar_y.min(), bar_y.max())
    axins.set_xlim(1, 2)
    axins.grid(False)
    axins.set_facecolor('white')
    axins.set_xticks([])
    axins.set_yticks(np.linspace(min(bar_y), max(bar_y), 5))
    axins.yaxis.tick_right()
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax1.set_title('A.', fontsize=24, loc='left', y=1, x=-.1)
    ax3.set_title('B.', fontsize=24, loc='left', y=1, x=-.1)
    ax5.set_title('C.', fontsize=24, loc='left', y=1, x=-.1)
    ax7.set_title('D.', fontsize=24, loc='left', y=1, x=-.1)
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3, left=True, right=False)
    sns.despine(ax=ax4)
    sns.despine(ax=ax5)
    sns.despine(ax=ax6)
    sns.despine(ax=ax7)
    center_ylabels(ax3, ax4)
    plt.savefig(os.path.join(fig_path, filename[:-4] + '.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, filename[:-4] + '.svg'),
                bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, filename[:-4] + '.png'),
                dpi=600, bbox_inches='tight')
