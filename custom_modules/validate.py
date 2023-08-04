import re
from collections import defaultdict
from typing import List, Tuple, Dict
from inspect import signature
from pydoc import locate
########################################################################

def err(error:str, param, passed=None, accepted=None):
    if error == 'TypeError':
        dt = type(passed).__name__
        raise TypeError(f'Invalid type \'{dt}\' found in parameter \'{param}\'. {param} must be of type: {accepted}.')
    elif error == 'ValueError':
        raise ValueError(f'Invalid \'{passed}\' value found in parameter \'{param}\'. {param} = {accepted}.')
    elif error == 'MissingArgument':
        raise TypeError(f'Missing argument {param}.')
    elif error == 'UnknownArgument':
        raise TypeError(f'Unknown argument {param}.')
    elif error == 'KeyError':
        raise KeyError(f'{param}')
########################################################################

def unpack(string, _dts=[]):
    pattern = "([\[\({].*[\]\)}])"
    dt = re.split(pattern, string)
    dt = list(filter(None, dt))

    if len(dt) == 1:
        dt0 = dt.pop(0).strip()
        dt0 = dt0.split(',')
        if len(dt0) == 1:
            _dts.append(dt0.pop(0).strip().replace('[', '').replace(']', ''))
        if len(dt0) > 1:
            for i in dt0:
                unpack(i, _dts)
    else:
        for i in dt:
            unpack(i.strip(), _dts)

    return _dts
########################################################################
'''
def valid(val, dtypes:list) -> bool:
    res = True
    if dtypes:
        dt = dtypes[0]

        if not isinstance(val, dt):
            res = False
    
        elif isinstance(val, list) or isinstance(val, tuple) or isinstance(val, set):
            for v in val:
                res = valid(v, dtypes[1:])

        elif isinstance(val, dict):
            for k in val.keys():
                res = valid(k, dtypes[1:])
                if dtypes:
                    res = valid(val[k], dtypes[2:])
    return res
'''


def valid(val, dtypes:list, _res=[True]) -> bool:
    if not _res[0]: return _res

    if dtypes:
        dt = dtypes[0]

        if not isinstance(val, dt):
            _res = [False, val, str(dt)]
    
        elif isinstance(val, list) or isinstance(val, tuple) or isinstance(val, set):
            for v in val:
                _res = valid(v, dtypes[1:], _res)

        elif isinstance(val, dict):
            for k in val.keys():
                _res = valid(k, dtypes[1:], _res)
                if dtypes:
                    _res = valid(val[k], dtypes[2:], _res)
    return _res
########################################################################

def validate(options:dict={}):
    def inner(func):
        def wrapper(*args, **kwargs):
            # collect accepted args dtypes
            dtypes = defaultdict()
            dtypes_string = defaultdict()
            # keep track of args that accept a default to avoid raising error if missing
            defaults = []

            # get function signature
            sig = signature(func).parameters

            # parse function signature
            for k in sig.keys():
                p = str(sig[k]).split(':')

                # if len > 1 after split on ':' -> means dtype available in function signature for this arg
                if len(p) > 1:
                    dt = p[1].split('=', 1)

                    # if len > 1 after split on '=' -> means accepts a default
                    if len(dt) > 1:
                        defaults.append(k)

                    dt = dt[0].strip()

                    # check if dtype is composed of 2 dtypes. e.g. List[int], set(str)

                    dtypes_string[p[0].strip()] = dt

                    dt = unpack(dt, [])
                    dt2 = []
                    for d in dt:
                        try:
                            dt2.append(eval(d))
                        except:
                            dt2.append(locate(d))
                    dt = dt2

                    dtypes[p[0].strip()] = dt


                # no dtype detected in function signature
                else:
                    dtypes[p[0].strip()] = None

            # collect passed args with their values
            passed = defaultdict()
            dtypes_keys = list(dtypes.keys())

            # parse kwargs
            for k in kwargs:
                # passed kwarg that is not in function signature
                if k not in dtypes_keys: err('UnknownArgument', k)
                passed[k] = kwargs[k]
            # parse args
            for i in range(len(args)):
                passed[dtypes_keys[i]] = args[i]

            # NOTE remove and keep built-in handling
            # check for missing args
            for k in dtypes:
                if (dtypes[k] != None) and (k not in defaults):
                    if k not in passed.keys(): err('MissingArgument', k)

            # validate passed args dtypes against function signature
            # also validate passed args values if a valid list is available

            for k in passed:
                if dtypes[k] != None:
                    check = valid(passed[k], dtypes[k])
                    if not check[0]:
                        err('TypeError', k, check[1], dtypes_string[k].split('.')[-1])
                        #err('TypeError', k, passed[k], dtypes_string[k].split('.')[-1])
                if k in options.keys():
                    if passed[k] not in options[k]: err('ValueError', k, passed[k], options[k])

            return func(*args, **kwargs)
        return wrapper
    return inner
########################################################################


# import pandas as pd

# @validate()
# def plot_sector(dataframe:pd.core.frame.DataFrame, year:int, figsize:Tuple[int]=(16, 9), show:bool=True, save_fig:bool=False):
#     pass

# df = pd.DataFrame({'key':['value']})
# plot_sector(df, year=2018, figsize=('s', 8), show=False, save_fig=False)






