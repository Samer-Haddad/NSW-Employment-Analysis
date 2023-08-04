import matplotlib as mplot
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

THEME = plt.rcParams['axes.prop_cycle'].by_key()['color']

COLORS ={
        'Public Sector':THEME[1],
        'PT':THEME[4],
        'PT/Ratio':'#ffffff',
        'FT':THEME[2],
        'F':THEME[0],
        'M':THEME[3],
        'PT/M':'#61d8e8',
        'PT/M/Ratio':'#72a382',
        'PT/F':'#9545b0',
        'PT/F/Ratio':'#9872a3',
        'FT/M':'#245091',
        'FT/F':'#8a1333',
        'WF':THEME[1],
        'Education':'#e04646',
        'Family & Community Services':'#f2914b',
        'Finance, Services & Innovation':'#e3bd32',
        'Health':'#b6e635',
        'Industry':'#2fe085',
        'Justice':'#34e1eb',
        'Planning & Environment':'#3e79e6',
        'Premier & Cabinet':'#b446eb',
        'Transport':'#f23ac4',
        'Treasury':'#d16d6d',
        }
########################################################################

def load_theme():
    '''
    Download and use Rose-Pine theme from github
    '''
    styles_dir = Path.joinpath(Path(mplot.get_configdir()), 'stylelib')
    style_path = Path.joinpath(styles_dir, 'rose-pine.mplstyle')

    if not Path.exists(styles_dir):
        styles_dir.mkdir(parents=True, exist_ok=True)

    if Path.is_file(style_path):
        mplot.style.reload_library()
        plt.style.use('rose-pine')
    
    else:
        directory = 'rose-pine-matplotlib/themes'
        if not Path.exists('rose-pine-matplotlib'):
            try:
                _ = subprocess.run(["git", "clone", "https://github.com/h4pZ/rose-pine-matplotlib"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            except:
                pass

        try: 
            for f in os.listdir(directory):
                src = Path.joinpath(directory, f)
                dst = Path.joinpath(styles_dir, f)
                shutil.copy2(src, dst)
            mplot.style.reload_library()
            plt.style.use('rose-pine')
        except:
            pass

    THEME = plt.rcParams['axes.prop_cycle'].by_key()['color']
    COLORS['Public Sector'] = THEME[1]
    COLORS['PT'] = THEME[4]
    COLORS['FT'] = THEME[2]
    COLORS['F'] = THEME[0]
    COLORS['M'] = THEME[3]
    COLORS['WF'] = THEME[1]
    return THEME, COLORS
########################################################################

def autopct_format(values):
    '''
    Format percentage and value text on plots
    '''
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:,})'.format(pct, v=val)
    return my_format
########################################################################

def wrap_labels(ax, width, rot=0, break_long_words=False, xaxis=True):
    '''
    Wrap label text and avoid overlapping labels on plots
    '''
    labels = []
    if xaxis:
        ticklabels = ax.get_xticklabels()
    else:
        ticklabels = ax.get_yticklabels()
    for label in ticklabels:
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    if xaxis:
        ax.set_xticklabels(labels, rotation=rot)
    else:
        ax.set_yticklabels(labels, rotation=rot)
########################################################################