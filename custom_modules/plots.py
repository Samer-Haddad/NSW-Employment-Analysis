import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Tuple, Dict, List
from statsmodels.tsa.stattools import adfuller
from .style import autopct_format, wrap_labels, load_theme
from .stats import workforce_total, workforce_gender, CAT, STAT
from .validate import validate, err
from .stats import statistics

DIR_STATISTICS = Path('Images/Statistics/')
DIR_FORECAST = Path('Images/Forecast/')
########################################################################

@validate()
def check_dir(path:Path):
    if not Path.exists(path):
        path.mkdir(parents=True, exist_ok=True)
########################################################################

@validate()
def plot_sector(dataframe:pd.core.frame.DataFrame, year:int, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Total Part-Time Employent in the public sector
    Takes the raw dataframe as input.
    '''

    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
    THEME, _ = load_theme()
    dfx = workforce_total(dataframe, year)
    df_pt, df_ft = dfx['pt'], dfx['ft']

    # Total Part-Time Employment in the Public Sector
    hc_pt = df_pt['Headcount'].sum()
    hc_ft = df_ft['Headcount'].sum()
    title = f'({year}) Part-Time Employment in the Public Sector'
    y = np.array([hc_pt, hc_ft])
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'aspect':'equal'})
    _, _, autotexts = ax.pie(y, explode=[0.05, 0], shadow=False, autopct=autopct_format(y), textprops={'fontsize': 16}) #labels = ['Part-Time Employees', 'Full-Time Employees'], 
    for autotext in autotexts:
        autotext.set_color('#253652')
        autotext.set_fontsize(20)
        autotext.set_fontweight('extra bold')

    leg = [Line2D([0], [0], color=THEME[0], lw=8), Line2D([0], [0], color=THEME[1], lw=8), Line2D([0], [0], color=THEME[3], lw=8)]


    fig.legend(leg, ['Part-Time', 'Full-Time'], loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.title(title, pad=10, fontsize=20)
    if show: plt.show()
    if save_fig:
        check_dir(DIR_STATISTICS)
        save_title = f'{title}.png'
        path = Path.joinpath(DIR_STATISTICS, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)

########################################################################

@validate()
def plot_cluster(dataframe:pd.core.frame.DataFrame, year:int, figsize:Tuple[int]=(16, 9), sort:bool=True, h:bool=False, show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Part-Time Employent per Cluster
    Takes the raw dataframe as input.
    enable h for horizontal bar chart
    '''
    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
    THEME, _ = load_theme()
    dfx = workforce_total(dataframe, year)
    df_cluster = dfx['cluster']
    if sort:
        df_cluster.sort_values(['PT'], ascending=False, inplace=True)

    idx = np.array(df_cluster['Cluster'])
    pt = np.array(df_cluster['PT'])
    w = 0.25

    fig, ax = plt.subplots(figsize = figsize)

    if not h:
        '''
        Plot bar graph.
        comparison between part-time and full-time.
        '''
        title = f'({year}) Part-Time vs. Full-Time'
        ft = np.array(df_cluster['FT'])
        pos1 = np.arange(len(pt))
        pos2 = [x + w for x in pos1]

        ax.bar(pos1, pt, width = w, label ='Part-Time')
        ax.bar(pos2, ft, width = w, label ='Full-Time')
        ax.set_xticks([x + w/2 for x in range(len(pt))])
        ax.set_xticklabels(idx, fontsize=16)

        wrap_labels(ax, width=10)
        plt.xlabel('Cluster', labelpad=25, fontsize=20)
        plt.ylabel('Headcount', labelpad=25, fontsize=20)

    else:
        '''
        Plot horizontal bar graph.
        Not a comparison between part-time and full-time.
        shows part-time percentage of each cluster.
        '''
        title = f'({year}) Part-Time Employment per Cluster'
        wf = np.array(df_cluster['WF'])
        pos = np.arange(len(pt))

        ax.barh(pos, pt, label ='Part-Time', zorder=2)
        ax.barh(pos, wf, label ='Total Workforce', zorder=1)
        
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 10)

        ax.grid(visible=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
        ax.invert_yaxis()

        ax.set_yticks([x + w/4 for x in range(len(pt))])
        ax.set_yticklabels(idx, fontsize=12)

        for i, v in enumerate(pt):
            space = np.log10(v)*1000
            space = 0 if (v + space) < wf[i] else wf[i] # v + 1000 + space
            ax.text(wf[i], i + 0.12, str(f'{((v/wf[i])*100):.0f}%'), color='#253652', fontweight='bold', fontstyle='italic', fontsize=14, bbox=dict(facecolor=THEME[0], edgecolor=THEME[0], pad=1))  #  'white'   '#7b95bd'

        wrap_labels(ax, width=10, rot=30, xaxis=False)
        plt.xlabel('Headcount', labelpad=25, fontsize=20)
        plt.ylabel('Cluster', labelpad=25, fontsize=20)
        
    plt.tight_layout()
    plt.legend()
    plt.title(title, pad=10, fontsize=20)
    if show: plt.show()
    if save_fig:
        check_dir(DIR_STATISTICS)
        save_title = f'{title}.png'
        path = Path.joinpath(DIR_STATISTICS, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
########################################################################

@validate()
def plot_sector_gender_bar(dataframe:pd.core.frame.DataFrame, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Plot both trends on same scale
    Takes the raw dataframe as input.
    '''
    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
    THEME, _ = load_theme()

    # Split data into male and female
    df_female = dataframe[dataframe['Gender']=='Female']
    df_male = dataframe[dataframe['Gender']=='Male']

    df_female = df_female.groupby(['Year'])[['Headcount']].sum().reset_index().rename(columns={'Headcount': 'F'})
    df_male = df_male.groupby(['Year'])[['Headcount']].sum().reset_index().rename(columns={'Headcount': 'M'})

    df_gender = df_female.merge(df_male, left_on='Year', right_on='Year')

    x_axis = np.array(df_gender['Year'])
    y_male = np.array(df_gender['M'])
    y_female = np.array(df_gender['F'])

    fig, _ = plt.subplots(figsize=figsize)
    plt.plot(x_axis, y_male, color=THEME[3])
    plt.plot(x_axis, y_female, color=THEME[0])

    plt.bar(x_axis - 0.2, y_male, 0.4, label = 'Male Headcount', color=THEME[3]) #'#1e76d4'
    plt.bar(x_axis + 0.2, y_female, 0.4, label = 'Female Headcount', color=THEME[0])

    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1), fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Male vs. Female Employment', pad=50, fontsize=20)
    plt.xlabel('Year', labelpad=25, fontsize=20)
    plt.ylabel('Heacount', labelpad=25, fontsize=20)
    fig.legend(loc='outside right upper')
    if show: plt.show()
    if save_fig:
        check_dir(DIR_STATISTICS)
        save_title = 'Male vs. Female Employment.png'
        path = Path.joinpath(DIR_STATISTICS, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
########################################################################

@validate()
def plot_sector_gender(dataframe:pd.core.frame.DataFrame, year:int, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Gender Part-Time Employment in the Public Sector
    Takes the raw dataframe as input.
    '''
    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
    THEME, _ = load_theme()
    dfx = workforce_gender(dataframe, year)
    df_pt_male, df_ft_male, df_pt_female, df_ft_female = dfx['pt_m'], dfx['ft_m'], dfx['pt_f'], dfx['ft_f']

    hc_pt_male = df_pt_male['Headcount'].sum()
    hc_ft_male = df_ft_male['Headcount'].sum()
    hc_pt_female = df_pt_female['Headcount'].sum()
    hc_ft_female = df_ft_female['Headcount'].sum()

    title = f'({year}) Gender Part-Time Employment in the Public Sector'
    y1 = np.array([hc_pt_male, hc_ft_male])
    y2 = np.array([hc_pt_female, hc_ft_female])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, subplot_kw={'aspect':'equal'})
    _, _, autotexts = ax1.pie(y1, colors=[THEME[3], THEME[1]], explode=[0.05, 0], shadow=False, autopct=autopct_format(y1), textprops={'fontsize': 16}) #labels = ['Part-Time Male', 'Full-Time Male'], 
    for autotext in autotexts:
        autotext.set_color('#253652')
        autotext.set_fontsize(20)
        autotext.set_fontweight('bold')
    #ax1.set_title('Male Employment').set_fontsize(20)

    _, _, autotexts = ax2.pie(y2, explode=[0.05, 0], shadow=False, autopct=autopct_format(y2), textprops={'fontsize': 16}) #labels = ['Part-Time Female', 'Full-Time Female'], 
    for autotext in autotexts:
        autotext.set_color('#253652')
        autotext.set_fontsize(20)
        autotext.set_fontweight('bold')
    #ax2.set_title('Female Employment').set_fontsize(20)

    
    leg = [Line2D([0], [0], color=THEME[1], lw=8), Line2D([0], [0], color=THEME[0], lw=8), Line2D([0], [0], color=THEME[3], lw=8)]
    
    fig.suptitle(title, y=0.95, x=0.5, fontsize=20) # y = 1
    fig.legend(leg, ['Gender Full-Time', 'Female Part-Time', 'Male Part-Time'], loc='upper right', fontsize=14)
    plt.tight_layout()
    if show: plt.show()
    if save_fig:
        check_dir(DIR_STATISTICS)
        save_title = f'{title}.png'
        path = Path.joinpath(DIR_STATISTICS, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
########################################################################

@validate()
def plot_cluster_gender(dataframe:pd.core.frame.DataFrame, year:int, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Gender Part-Time Employment per Cluster
    Takes the raw dataframe as input.
    '''
    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
    THEME, _ = load_theme()

    dfx = workforce_gender(dataframe, year)
    df_cluster = dfx['cluster']

    y_axis = np.array(df_cluster['Cluster'])
    x_axis_male = np.array(df_cluster['M'])
    x_axis_female = np.array(df_cluster['F'])
    x_axis_male_pt = np.array(df_cluster['PT/M'])
    x_axis_female_pt = np.array(df_cluster['PT/F'])

    title = f'({year}) Gender Employment per Cluster'
    xticks = np.arange(20000, 120000, 20000)

    fig, axes = plt.subplots(figsize=figsize, ncols=2, sharey=True)
    fig.tight_layout()
    
    # Male plot (on the left)
    axes[0].barh(y_axis, x_axis_male_pt, align='center', zorder=2, color=THEME[3], label ='Male Part-Time (%)') #'#1e76d4'
    axes[0].barh(y_axis, x_axis_male, align='center', zorder=1, color=THEME[1], label ='Gender Total Workforce')
    axes[0].set_title('Male', fontsize=20, pad=15)
    axes[0].set(xticks=xticks, xticklabels=xticks)
    axes[0].set(yticks=y_axis, yticklabels=y_axis)
    #axes[0].set_xlim(xmax = 50000)
    axes[0].legend(loc='lower left')
    wrap_labels(axes[0], width=20, rot=0, xaxis=False)

    # Female plot (on the right)
    axes[1].barh(y_axis, x_axis_female_pt, align='center', zorder=2, color=THEME[0], label ='Female Part-Time (%)')
    axes[1].barh(y_axis, x_axis_female, align='center', zorder=1, color=THEME[1], label ='Gender Total Workforce')
    axes[1].set_title('Female', fontsize=20, pad=15)
    axes[1].set(xticks=xticks, xticklabels=xticks)
    axes[1].legend(loc='lower right')

    # Display percentage values
    # for i, v in enumerate(x_axis_male_pt):
    #     axes[0].text(v+8000, i + 0.12, str(f'{((v/x_axis_male[i])*100):.0f}%'), color='white', fontweight='bold', fontstyle='italic', fontsize=14)
    # for i, v in enumerate(x_axis_female_pt):
    #     axes[1].text(v+2000, i + 0.12, str(f'{((v/x_axis_female[i])*100):.0f}%'), color='white', fontweight='bold', fontstyle='italic', fontsize=14)

    for i, v in enumerate(x_axis_male_pt):
        axes[0].text(x_axis_male[i]+5000, i + 0.12, str(f'{((v/x_axis_male[i])*100):.0f}%'), color=THEME[3], fontweight='bold', fontstyle='italic', fontsize=14)#, bbox=dict(facecolor=THEME[3], edgecolor=THEME[3], pad=1)) #'#253652'
    for i, v in enumerate(x_axis_female_pt):
        axes[1].text(x_axis_female[i]+1000, i + 0.12, str(f'{((v/x_axis_female[i])*100):.0f}%'), color=THEME[0], fontweight='bold', fontstyle='italic', fontsize=14)#, bbox=dict(facecolor=THEME[0], edgecolor=THEME[0], pad=1))

    axes[0].yaxis.tick_left()
    axes[0].tick_params(axis='y', labelsize=14) #colors='white', 
    axes[0].invert_xaxis()
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.005, top=0.85, bottom=0.1, left=0.18, right=0.95)
    fig.suptitle(title, y=1, x=0.5, fontsize=20)

    if show: plt.show()
    if save_fig:
        check_dir(DIR_STATISTICS)
        save_title = f'{title}.png'
        path = Path.joinpath(DIR_STATISTICS, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
########################################################################

@validate()
def plot_change(dataframe:pd.core.frame.DataFrame, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Male and Female Part-Time Employent change (%)
    Takes the raw dataframe as input.
    '''
    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')

    THEME, _ = load_theme()
    title = f'Part-Time Representation Change (since 2014)'

    stats = statistics(dataframe)
    
    x_ticks = [x for x in range(len(list(stats.keys())))]
    x_ticklabels = np.array([x for x in stats])
    y_ticks = np.arange(-0.2, 0.25, 0.05)
    y_ticklabels = [f'{float(x*100):.0f}%' for x in y_ticks]
    
    part_time = []
    part_time_male = []
    part_time_female = []

    for cat in stats.keys():
        pt = stats[cat]['PT/Ratio']
        pt_m = stats[cat]['PT/M/Ratio']
        pt_f = stats[cat]['PT/F/Ratio']
        part_time.append(pt[-1]-pt[0])
        part_time_male.append(pt_m[-1]-pt_m[0])
        part_time_female.append(pt_f[-1]-pt_f[0])

    fig, ax = plt.subplots(figsize = figsize)

    ax.bar(x_ticks, part_time, width = 0.60, label ='PT Change', color=THEME[1]) #Theme[1]
    ax.bar(x_ticks, part_time_male, width = 0.40, label ='Male PT Change', color=THEME[3]) #Theme[3]
    ax.bar(x_ticks, part_time_female, width = 0.20, label ='Female PT Change', color=THEME[0]) #Theme[0]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=16)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=16)

    wrap_labels(ax, width=10)
    #plt.xlabel('Cluster', labelpad=25, fontsize=20)
    plt.ylabel('Percentage', labelpad=25, fontsize=20)
        
    plt.tight_layout()
    plt.legend()
    plt.title(title, pad=10, fontsize=20)
    if show: plt.show()
    if save_fig:
        check_dir(DIR_STATISTICS)
        save_title = f'{title}.png'
        path = Path.joinpath(DIR_STATISTICS, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
########################################################################

@validate()
def plot_stats(dataframe:pd.core.frame.DataFrame, lines:Dict[str, List[str]], title:str=None, double_scale:bool=False, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Unlike previous functions, this one takes the transpose output of the statistics functions, not the raw dataframe.
    lines = dict(list()) | e.g. lines = {'Public Sector':['PT', 'FT'], 'Education':['PT/M']}
    title = [None, 'auto', str] | if title='auto' then plot title will be auto generated for up to 4 lines
    Enabling "double_scale" will plot 2 lines on the same x-axis but on separate scales (y-axis). (Does not work for more than 2 lines)
    '''

    for k in lines:
        if k not in CAT:
            err('KeyError', k)
        for v in lines[k]:
            if v not in STAT:
                err('KeyError', v)

    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
    
    _, COLORS = load_theme()

    save_title = title
    #stats = statistics(dataframe)

    plt.ioff()
    x_axis = np.array(dataframe.index.to_list())
    #x_axis = np.array(dataframe['Year'].unique())
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.set(xticks=x_axis, xticklabels=x_axis)
    ax.set_xticklabels(x_axis, fontsize=14)

    keys = list(lines.keys())
    values = [sorted(set(x)) for x in lines.values()]
    
    perm = []
    for i in range(len(keys)):
        for pair in list(zip([keys[i]]*len(values[i]), values[i])):
            perm.append(pair)

    nlines = len(perm)
    
    if nlines < 5:
        same_category = len(keys) == 1
        same_stats = values.count(values[0]) == len(values)

        if same_category or same_stats:
            title_auto = STAT[values[0][0]]
            for i in values[0][1:]:
                title_auto += ' vs. ' + STAT[i]
            title_auto += '\n[' + keys[0]
            for i in keys[1:]:
                title_auto += ', ' + i
            title_auto += ']'

        else:
            cat, stat = perm[0]
            title_auto = cat + ' ' + STAT[stat]
            for p in perm[1:]:
                cat, stat = p
                title_auto += ' vs. ' + cat + ' ' + STAT[stat]
    else:
        title_auto = 'Custom Plot'

    if title == 'auto':
        title = title_auto
        save_title = title.replace('\n', ' ')


    if (not double_scale) or (nlines != 2):
        for p in perm:
            cat, stat = p
            col = cat + '/' + stat
            lbl = cat + ' ' + STAT[stat]
            ax.plot(x_axis, dataframe[col], label=lbl, color=COLORS[stat], marker='o')
            #ax.set_yticklabels(dataframe[col], fontsize=12)
        fig.legend()
        y_label = 'Headcount'
        if stat.split('/')[-1].strip() == 'Ratio':
            y_label = 'Percentage'
        plt.ylabel(y_label, labelpad=25, fontsize=20)
        plt.xlabel('Year', labelpad=25, fontsize=20)
        
        

            
    elif (double_scale) and (nlines == 2):
        cat1, stat1 = perm[0]
        cat2, stat2 = perm[1]
        col1 = cat1 + '/' + stat1
        col2 = cat2 + '/' + stat2
        if stat1 in ['PT/Ratio', 'PT/M/Ratio', 'PT/F/Ratio']:
            cat1, stat1, col1, cat2, stat2, col2 = cat2, stat2, col2, cat1, stat1, col1

        y1 = dataframe[col1]
        label1 = cat1 + ' ' + STAT[stat1]
        color1 = COLORS[stat1] if stat1 != stat2 else COLORS[cat1]

        ax.plot(x_axis, y1, label=label1, color=color1, marker='o')
        ax.set_ylabel(STAT[stat1], labelpad=25, fontsize=20, color=color1)
        ax.set_xlabel('Year', labelpad=25, fontsize=20)
        if stat1 in ['PT/Ratio', 'PT/M/Ratio', 'PT/F/Ratio']:
            ax.set(yticks=y1, yticklabels=[f'{x*100:.1f}%' for x in y1])
            #ax.set_yticklabels([f'{x*100:.1f}%' for x in y1], fontsize=12)
        # else:
        #     ax.set(yticks=y1)
        #     ax.set_yticklabels(y1, fontsize=12)

        ax.tick_params(axis='y', labelcolor=color1)
        ax.set_xticklabels(x_axis, fontsize=14)
        
        y2 = dataframe[col2]
        label2 = cat2 + ' ' + STAT[stat2]
        color2 = COLORS[stat2] if stat1 != stat2 else COLORS[cat2]
        ax2 = ax.twinx()
        ax2.plot(x_axis, y2, label=label2, color=color2, marker='o')
        ax2.set_ylabel(STAT[stat2], labelpad=25, fontsize=20, color=color2)
        if stat2 in ['PT/Ratio', 'PT/M/Ratio', 'PT/F/Ratio']:
            ax2.set(yticks=y2, yticklabels=[f'{x*100:.1f}%' for x in y2])
            #ax2.set_yticklabels([f'{x*100:.1f}%' for x in y2], fontsize=12)
        # else:
        #     ax2.set(yticks=y2)
        #     ax2.set_yticklabels(y2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)

        #ax2.set_xticklabels(x_axis, fontsize=16)
        

    #fig.legend(loc='outside right lower')
    if title: plt.title(title, pad=50, fontsize=20)
    if show: plt.show()
    if save_fig:
        check_dir(DIR_STATISTICS)
        save_title = f'{save_title}.png'
        path = Path.joinpath(DIR_STATISTICS, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
    plt.close(fig)
########################################################################

@validate()
def plot_trend(df_history:pd.core.frame.DataFrame, df_forecast:pd.core.frame.DataFrame, cat:str, stat:str, title:str=None, figsize:Tuple[int]=(16, 9), show:bool=False, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Similar to plot_stats, this function takes the transpose of the statistics and the forecast dataframes.
    '''
    col = cat + '/' + stat
    if col not in df_history.columns.to_list():
        err('KeyError', col)
    if col not in df_forecast.columns.to_list():
        err('KeyError', col)
    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')

    THEME, _ = load_theme()
    save_title = title

    hist = df_history[col]
    fc = [hist.iloc[-1]]
    fc = fc + df_forecast[col].to_list()
    hist_years = df_history.index.to_list()
    fc_years = [hist_years[-1]] + df_forecast.index.to_list()

    x_label, y_label = 'Year', 'Headcount'
    x_ticks = hist_years + fc_years
    y_ticks = df_history[col].to_list() + df_forecast[col].to_list()
    y_ticklabels = y_ticks
    if stat.split('/')[-1].strip() == 'Ratio':
        y_label = 'Percentage'
        step = (y_ticks[-1]-y_ticks[0])/6
        y_ticks = np.arange(y_ticks[0], y_ticks[-1]+step, step)
        y_ticklabels = [f'{float(x*100):.1f}%' for x in y_ticks]

    plt.ioff()
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.set(xticks=x_ticks, xticklabels=x_ticks)

    ax.plot(hist_years, hist, label='History', color=THEME[3], marker='o')
    ax.plot(fc_years, fc, label='Forecast', color=THEME[0], linestyle='dashed', marker='o')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=14)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=14)

    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)

    plt.ylabel(y_label, labelpad=25, fontsize=20)
    plt.xlabel(x_label, labelpad=25, fontsize=20)

    fig.legend()
    if not title:
        title = f'Trend Projection\n{STAT[stat]}\n[{cat}]'
        save_title = f'Trend Projection {STAT[stat]} [{cat}]'
    plt.title(title, pad=50, fontsize=20)
    if show: plt.show()
    if save_fig:
        check_dir(DIR_FORECAST)
        save_title = f'{save_title}.png'.replace('/', '-')
        path = Path.joinpath(DIR_FORECAST, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
    plt.close(fig)
########################################################################

@validate()
def plot_forecast_change(history:dict, forecast:dict, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Takes statistics dictionary and forecast dictionary. Not the trasnpose.
    '''
    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')

    THEME, _ = load_theme()
    title = f'Part-Time Representation Change Forecast (by 2025)'
    
    x_ticks = [x for x in range(len(list(history.keys())))]
    x_ticklabels = np.array([x for x in history])
    y_ticks = np.arange(-0.2, 0.9, 0.1)
    y_ticklabels = [f'{float(x*100):.0f}%' for x in y_ticks]
    
    part_time = []
    part_time_male = []
    part_time_female = []

    for cat in history.keys():
        hist_pt, fc_pt = history[cat]['PT/Ratio'], forecast[cat]['PT/Ratio']
        hist_pt_m, fc_pt_m = history[cat]['PT/M/Ratio'], forecast[cat]['PT/M/Ratio']
        hist_pt_f, fc_pt_f = history[cat]['PT/F/Ratio'], forecast[cat]['PT/F/Ratio']
        part_time.append(fc_pt[-1]-hist_pt[-1])
        part_time_male.append(fc_pt_m[-1]-hist_pt_m[-1])
        part_time_female.append(fc_pt_f[-1]-hist_pt_f[-1])

    fig, ax = plt.subplots(figsize = figsize)

    ax.bar(x_ticks, part_time, width = 0.60, label ='PT Change', color=THEME[1]) #Theme[1]
    ax.bar(x_ticks, part_time_male, width = 0.40, label ='Male PT Change', color=THEME[3]) #Theme[3]
    ax.bar(x_ticks, part_time_female, width = 0.20, label ='Female PT Change', color=THEME[0]) #Theme[0]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=16)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=16)

    wrap_labels(ax, width=10)
    #plt.xlabel('Cluster', labelpad=25, fontsize=20)
    plt.ylabel('Percentage', labelpad=25, fontsize=20)
        
    plt.tight_layout()
    plt.legend()
    plt.title(title, pad=10, fontsize=20)
    if show: plt.show()
    if save_fig:
        check_dir(DIR_FORECAST)
        save_title = f'{title}.png'
        path = Path.joinpath(DIR_FORECAST, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
########################################################################

@validate()
def stationary_test(dataframe:pd.core.frame.DataFrame, cat:str, stat:str, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False, dpi:int=300, transparent:bool=False):
    '''
    Similar to plot_stats, this function takes the transpose output of the statistics function.
    '''
    if len(figsize) != 2:
        err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
    THEME, _ = load_theme()
    col = cat + '/' + stat
    lbl = cat + ' ' + STAT[stat]
    ma = dataframe[col].rolling(3).mean()
    std = dataframe[col].rolling(3).std()
    
    plt.ioff()
    x_axis = np.array(dataframe.index.tolist())
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.set(xticks=x_axis, xticklabels=x_axis)
    ax.set_xticklabels(fontsize=16)
    ax.set_yticklabels(fontsize=16)
    ax.plot(x_axis, dataframe[col], label=lbl, color=THEME[3], marker='o')
    ax.plot(x_axis, ma, label='Rolling Mean', color=THEME[1], marker='o')
    ax.plot(x_axis, std, label ='Rolling Std', color=THEME[0], marker='o')
    ax.legend()
    plt.title(f'Moving Average & Standard Deviation\n[{lbl}]', pad=50, fontsize=20)

    if show: plt.show()
    if save_fig:
        save_title = f'{lbl} [MASD].png'
        check_dir(DIR_STATISTICS)
        path = Path.joinpath(DIR_STATISTICS, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
    plt.close(fig)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(dataframe[col], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput.to_markdown())
########################################################################