import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List
from .validate import err, validate

CAT = ['Public Sector',
       'Education',
       'Family & Community Services',
       'Finance, Services & Innovation',
       'Health', 'Industry', 'Justice',
       'Planning & Environment',
       'Premier & Cabinet',
       'Transport',
       'Treasury'
       ]

STAT ={
        'PT':'Part-Time',
        'PT/Ratio':'Part-Time (%)',
        'FT':'Full-Time',
        'F':'Female',
        'M':'Male',
        'PT/M':'Part-Time Male',
        'PT/M/Ratio':'Part-Time Male (%)',
        'PT/F':'Part-Time Female',
        'PT/F/Ratio':'Part-Time Female (%)',
        'FT/M':'Full-Time Male',
        'FT/F':'Full-Time Female',
        'WF':'Total Workforce',
        }
########################################################################

@validate()
def workforce_total(dataframe:pd.core.frame.DataFrame, year:int) -> dict:
    '''
    Split into part-time and full-time of the given year
    returns a dictionary of 3 dataframes:
    pt: contains only part-time,
    ft: contains only full-time,
    cluster: contains pt and ft both grouped on cluster
    '''
    if 'Year' not in dataframe.columns.to_list():
        err('KeyError', 'Year')
    if year not in dataframe['Year'].to_list():
        err('KeyError', year)

    df = dataframe[dataframe['Year']==year]
    df_pt = df[df['PT/FT']=='Part-Time']
    df_ft = df[df['PT/FT']=='Full-Time']

    # Group by cluster
    df_pt_cluster = df_pt.groupby(['Cluster'])['Headcount'].sum().reset_index().rename(columns={'Headcount':'PT'})
    df_ft_cluster = df_ft.groupby(['Cluster'])['Headcount'].sum().reset_index().rename(columns={'Headcount':'FT'})
    df_cluster = df_pt_cluster.merge(df_ft_cluster, left_on='Cluster', right_on='Cluster')
    df_cluster['WF'] = df_cluster['FT'] + df_cluster['PT']

    return {'pt':df_pt, 'ft':df_ft, 'cluster':df_cluster}
########################################################################

@validate()
def workforce_gender(dataframe:pd.core.frame.DataFrame, year:int) -> dict:
    '''
    Similar to workforce_total(), but will split the data further based on gender.
    returns a dictionary of 5 dataframes:
    pt_f: part-time female
    ft_f: full-time female
    pt_m: part-time male
    ft_m: full-time male
    cluster: all the above grouped on cluster
    '''

    if 'Year' not in dataframe.columns.to_list():
        err('KeyError', 'Year')
    if year not in dataframe['Year'].to_list():
        err('KeyError', year)

    df = dataframe[dataframe['Year']==year]
    df_pt = df[df['PT/FT']=='Part-Time']
    df_ft = df[df['PT/FT']=='Full-Time']

    # Split into male and female part-time employees
    df_pt_female = df_pt[df_pt['Gender']=='Female']
    df_pt_male = df_pt[df_pt['Gender']=='Male']
    df_ft_female = df_ft[df_ft['Gender']=='Female']
    df_ft_male = df_ft[df_ft['Gender']=='Male']

    # Group by cluster
    df_pt_female_cluster = df_pt_female.groupby(['Cluster'])['Headcount'].sum().reset_index().rename(columns={'Headcount':'PT/F'})
    df_ft_female_cluster = df_ft_female.groupby(['Cluster'])['Headcount'].sum().reset_index().rename(columns={'Headcount':'FT/F'})
    df_pt_male_cluster = df_pt_male.groupby(['Cluster'])['Headcount'].sum().reset_index().rename(columns={'Headcount':'PT/M'})
    df_ft_male_cluster = df_ft_male.groupby(['Cluster'])['Headcount'].sum().reset_index().rename(columns={'Headcount':'FT/M'})

    df_female_cluster = df_pt_female_cluster.merge(df_ft_female_cluster, left_on='Cluster', right_on='Cluster')
    df_female_cluster['F'] = df_female_cluster['PT/F'] + df_female_cluster['FT/F']
    df_male_cluster = df_pt_male_cluster.merge(df_ft_male_cluster, left_on='Cluster', right_on='Cluster')
    df_male_cluster['M'] = df_male_cluster['PT/M'] + df_male_cluster['FT/M']

    df_cluster = df_female_cluster.merge(df_male_cluster, left_on='Cluster', right_on='Cluster')
    
    return {'pt_f':df_pt_female, 'ft_f':df_ft_female, 'pt_m':df_pt_male, 'ft_m':df_ft_male, 'cluster':df_cluster}
########################################################################

@validate()
def statistics(dataframe:pd.core.frame.DataFrame) -> dict:
    '''
    Returns a dictionary containing all relevant statistics
    '''

    if 'Year' not in dataframe.columns.to_list():
        err('KeyError', 'Year')

    stats = defaultdict(lambda: defaultdict(list))
    sector_stats = defaultdict(list)
    years = [int(x) for x in dataframe['Year'].unique()]

    for year in years:
        dfx_total = workforce_total(dataframe, year)
        dfx_gender = workforce_gender(dataframe, year)
        pt, ft, cluster = dfx_total['pt'], dfx_total['ft'], dfx_total['cluster']
        pt_f, ft_f, pt_m, ft_m, cluster_gender = dfx_gender['pt_f'], dfx_gender['ft_f'], dfx_gender['pt_m'], dfx_gender['ft_m'], dfx_gender['cluster']

        sector_stats['PT'].append(pt['Headcount'].sum())
        sector_stats['FT'].append(ft['Headcount'].sum())
        sector_stats['WF'].append(pt['Headcount'].sum()+ft['Headcount'].sum())
        sector_stats['PT/Ratio'].append(pt['Headcount'].sum()/(pt['Headcount'].sum()+ft['Headcount'].sum()))
        sector_stats['PT/F'].append(pt_f['Headcount'].sum())
        sector_stats['PT/F/Ratio'].append(pt_f['Headcount'].sum()/(pt_f['Headcount'].sum()+ft_f['Headcount'].sum()))
        sector_stats['FT/F'].append(ft_f['Headcount'].sum())
        sector_stats['F'].append(pt_f['Headcount'].sum()+ft_f['Headcount'].sum())
        sector_stats['PT/M'].append(pt_m['Headcount'].sum())
        sector_stats['PT/M/Ratio'].append(pt_m['Headcount'].sum()/(pt_m['Headcount'].sum()+ft_m['Headcount'].sum()))
        sector_stats['FT/M'].append(ft_m['Headcount'].sum())
        sector_stats['M'].append(pt_m['Headcount'].sum()+ft_m['Headcount'].sum())

        stats['Public Sector'] = sector_stats

        for i in range(len(cluster)):
            stats[cluster['Cluster'].iloc[i]]['PT'].append(cluster['PT'].iloc[i])
            stats[cluster['Cluster'].iloc[i]]['FT'].append(cluster['FT'].iloc[i])
            stats[cluster['Cluster'].iloc[i]]['WF'].append(cluster['WF'].iloc[i])
            stats[cluster['Cluster'].iloc[i]]['PT/Ratio'].append(cluster['PT'].iloc[i]/cluster['WF'].iloc[i])

        for i in range(len(cluster_gender)):
            stats[cluster_gender['Cluster'].iloc[i]]['PT/F'].append(cluster_gender['PT/F'].iloc[i])
            stats[cluster_gender['Cluster'].iloc[i]]['PT/F/Ratio'].append(cluster_gender['PT/F'].iloc[i]/(cluster_gender['PT/F'].iloc[i]+cluster_gender['FT/F'].iloc[i]))
            stats[cluster_gender['Cluster'].iloc[i]]['FT/F'].append(cluster_gender['FT/F'].iloc[i])
            stats[cluster_gender['Cluster'].iloc[i]]['F'].append(cluster_gender['F'].iloc[i])
            stats[cluster_gender['Cluster'].iloc[i]]['PT/M'].append(cluster_gender['PT/M'].iloc[i])
            stats[cluster_gender['Cluster'].iloc[i]]['PT/M/Ratio'].append(cluster_gender['PT/M'].iloc[i]/(cluster_gender['PT/M'].iloc[i]+cluster_gender['FT/M'].iloc[i]))
            stats[cluster_gender['Cluster'].iloc[i]]['FT/M'].append(cluster_gender['FT/M'].iloc[i])
            stats[cluster_gender['Cluster'].iloc[i]]['M'].append(cluster_gender['M'].iloc[i])

    return stats
########################################################################

@validate()
def transpose(dataframe:pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
    Transpose the dictionary or dataframe and unpack lists in rows
    '''
    tp = defaultdict(list)
    for k in dataframe:
        for k2 in dataframe[k]:
            col = k+'/'+k2
            tp[col] = dataframe[k][k2]
    return pd.DataFrame(tp)

@validate()
def transpose(dictionary:dict) -> pd.core.frame.DataFrame:
    tp = defaultdict(list)
    for k in dictionary:
        for k2 in dictionary[k]:
            col = k+'/'+k2
            tp[col] = dictionary[k][k2]
    return pd.DataFrame(tp)
########################################################################

@validate(options={'cat':CAT, 'stats':list(STAT.keys()), 'stats2':list(STAT.keys()), 'modes':['inter', 'intra', 'full', 'inspect'], 'sort':['corr', 'cat']})
def corr(dataframe:pd.core.frame.DataFrame, cat:List[str]=None, stats:List[str]=None, stats2:List[str]=None, modes:List[str]=['full'], t:float=0.7, sort:str='cat') -> pd.core.frame.DataFrame:
    '''
    Returns correlation pairs
    Takes the raw dataframe as input
    cat -> Category/Cluster, e.g. ['Public Sector', 'Education']
    stats -> which set of statistic to focus on, e.g. ['PT']
    stats2 -> if specific statistics required, e.g. ['FT']
    modes -> 4 different modes:
                inter: only pairs of different clusters
                intra: only pairs of the same cluster
                inspect: one of the pairs must contain a statistic from stats parameter
                full: everything (default)
    sort -> sort by category or correlation
    t = float(between 0.0 and 1.0), correlation threshold to be considered
    '''
    
    if t < 0.0 or t > 1.0:
        err('ValueError', 't', t, '[0.0 < t < 1.0]')

    df_statistics = statistics(dataframe)
    analysis = defaultdict(list)

    if not cat:
        cat = [x for x in CAT]
    if not stats:
        stats = [x for x in STAT.keys()]

    if stats2 or ('inspect' in modes):
        inspect = stats
        stats = [x for x in STAT.keys()]
    

    for k in df_statistics:
        if k in cat:
            for k2 in df_statistics[k]:
                if k2 in stats:
                    col = k+'/'+k2
                    analysis[col] = df_statistics[k][k2]

    analysis = pd.DataFrame(analysis)
    corr = analysis.corr()
    corr_pairs = corr.mask(np.tril(np.ones(corr.shape)).astype(bool))

    corr_pairs = corr_pairs[abs(corr_pairs) >= t].stack().reset_index()
    corr_pairs.columns = [*corr_pairs.columns[:-1], 'Correlation']

    def drop_row(row):
        lcat, lstat = row['level_0'].split('/', 1)[0], row['level_0'].split('/', 1)[1]
        rcat, rstat = row['level_1'].split('/', 1)[0], row['level_1'].split('/', 1)[1]
        for m in modes:
            if m == 'inter' and lcat == rcat: return True
            if m == 'intra' and (lcat != rcat or lstat == rstat) : return True
            if m == 'inspect' and all(s not in inspect for s in [lstat, rstat]): return True
            if stats2:
                if not ((lstat in inspect and rstat in stats2) or (lstat in stats2 and rstat in inspect)):
                    return True

    if stats2 or ('full' not in modes):
        for index, row in corr_pairs.iterrows():
            if drop_row(row):
                corr_pairs.drop(index, inplace=True)

    if sort == 'corr':
        corr_pairs = corr_pairs.reindex(corr_pairs['Correlation'].abs().sort_values(ascending=False).index)
    elif sort == 'cat':
        corr_pairs.sort_values(['level_0'], inplace=True)

    return corr_pairs
########################################################################