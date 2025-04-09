
import numpy as np
import pandas as pd
#from scipy import stats
import re 
import os
import warnings
warnings.filterwarnings('ignore')
import Tool
import bottleneck as bn
import process_expression
import json
args = json.load(open("./config.json", encoding="utf-8"))
CacheHandler = Tool.Handler(path = args['Handler_cache_path'],data_type = args['data_type'])
CacheHandler.reindex_like = CacheHandler['Close']
#CacheHandler.cashe_dict['mega_factor'] = pd.read_pickle('mega_factor_20250309.pkl')


def LS_cagr(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols):
    def CAGR(bt_ret):
        if not isinstance(bt_ret.index, pd.DatetimeIndex):
            bt_ret.index = pd.to_datetime(bt_ret.index)
        monthly_ret = bt_ret.resample('M').mean()
        n_months = (monthly_ret.index[-1].year - monthly_ret.index[0].year) * 12 + \
                   (monthly_ret.index[-1].month - monthly_ret.index[0].month)
        if n_months == 0:  # 避免除以零
            return -1e6
        return (abs(monthly_ret.add(1).prod()) ** (12 / n_months) - 1) * 100

    原始_expr = f'{individual}'
    try:
        total_length = parse_expression(f'{individual}', True)

        while (('SUE(' in f'{individual}') | ('Findelay(' in f'{individual}')):
            try:
                individual = process_expression.process_expression(f'{individual}')
            except:
                return -5000000, None

        pp_dict = preprocessing(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols)
        punish = pp_dict['punish']
        if punish:
            return punish, None

        yp = pp_dict['yp']
        target = pp_dict['target']

        LS_ret = (factor_to_weight(yp[target.notna()], True) * target).sum(axis=1)
        if LS_ret.notna().sum() / len(LS_ret) < 0.9:
            return -10, None

        mega_ret = pd.concat([scope['ts_result'], LS_ret], axis=1).mean(axis=1)

        score = abs(CAGR(mega_ret))
        score -= scope['mega_score']
        score -= total_length

        return score, LS_ret

    except Exception as e:
        raise ValueError(f'{原始_expr} 表达式运算失败:{e}')

def LS_cv(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols):
    原始_expr = f'{individual}'
    try:
        total_length = parse_expression(f'{individual}',True)
        #if total_length > 100:
        #    return -50,None
        while (('SUE(' in f'{individual}') | ('Findelay(' in f'{individual}')):
            try:
                individual = process_expression.process_expression(f'{individual}')
            except:
                return -50,None
        pp_dict = preprocessing(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols)
        punish = pp_dict['punish']
        if punish:return punish,None
        yp = pp_dict['yp']
        target = pp_dict['target']
        LS_ret = (factor_to_weight(yp[target.notna()],True) * target).sum(axis = 1)

        if LS_ret.notna().sum() / len(LS_ret) < 0.9:
            return -10,None
        LS_std = LS_ret.std()
        if LS_std == 0:
            punish -= 10
            score = np.nan
            return -10,None
        else:
            mega_ret = pd.concat([scope['ts_result'],LS_ret],axis  = 1).mean(axis=1)
            LS_mean = abs(mega_ret.mean())
            score = LS_mean / mega_ret.std() * 252**0.5
        score-=scope['mega_score']
        score-=(total_length)/1000
        return score,LS_ret
    except Exception as e:
        raise ValueError(f'{原始_expr} 表达式运算失败:{e}')

def factor_to_weight(alpha_array,only_long:bool = False):
    demeaned = alpha_array - np.nanmean(alpha_array,axis = 1)[:, None]
    weights = demeaned / np.nansum(np.abs(demeaned),axis = 1)[:, None]
    weights[np.isnan(weights)] = 0#检查数据无值时视为权重0
    if only_long:
        weights[weights<0] = 0
        weights*=2
    return weights

def IC(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols):
    原始_expr = f'{individual}'
    try:
        total_length = parse_expression(f'{individual}',True)
        #if total_length > 100:
        #    return -50,None
        while (('SUE(' in f'{individual}') | ('Findelay(' in f'{individual}')):
            try:
                individual = process_expression.process_expression(f'{individual}')
            except:
                return -50,None
        pp_dict = preprocessing(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols)
        punish = pp_dict['punish']
        if punish:return punish,None
        yp = pp_dict['yp']
        target = pp_dict['target']
        rank_IC = yp.corrwith(target, axis = 1, method = "spearman")
        if rank_IC.notna().sum() / len(rank_IC) < 0.9:
            return -10,None
        ic = rank_IC
        ic_mean = abs(ic.mean())
        score = 10*ic_mean
        #if total_length > 10:
        score-=(total_length)/1000
        return score,ic.sort_index()
    except Exception as e:
        raise ValueError(f'{原始_expr} 表达式运算失败:{e}')

def Score(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols):
    原始_expr = f'{individual}'
    try:
        total_length = parse_expression(f'{individual}',True)
        #if total_length > 100:
        #    return -50,None
        while (('SUE(' in f'{individual}') | ('Findelay(' in f'{individual}')):
            try:
                individual = process_expression.process_expression(f'{individual}')
            except:
                return -50,None
        pp_dict = preprocessing(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols)
        punish = pp_dict['punish']
        if punish:return punish,None
        yp = pp_dict['yp']
        target = pp_dict['target']
        rank_IC = yp.corrwith(target, axis = 1, method = "spearman")
        if rank_IC.notna().sum() / len(rank_IC) < 0.9:
            return -10,None
        ic = rank_IC
        ic_std = ic.std()
        if ic_std == 0:
            punish -= 10
            icir = np.nan
            return -10,None
        else:
            ic_mean = abs(ic.mean())
            icir = ic_mean / ic_std
        
        score = icir+10*ic_mean
        #if total_length > 10:
        score-=(total_length)/1000
        return score,ic.sort_index()
    except Exception as e:
        raise ValueError(f'{原始_expr} 表达式运算失败:{e}')
    
def Score_with_LS_CV(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols):
    原始_expr = f'{individual}'
    try:
        total_length = parse_expression(f'{individual}', True)

        while ('SUE(' in f'{individual}') or ('Findelay(' in f'{individual}'):
            try:
                individual = process_expression.process_expression(f'{individual}')
            except:
                return -50, None

        pp_dict = preprocessing(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols)
        punish = pp_dict['punish']
        if punish:
            return punish, None

        yp = pp_dict['yp']
        target = pp_dict['target']

        # ---- (1) 計算 rank IC 與 IR ----
        rank_IC = yp.corrwith(target, axis=1, method="spearman")
        if rank_IC.notna().sum() / len(rank_IC) < 0.9:
            return -10, None

        ic_mean = rank_IC.mean()
        ic_std = rank_IC.std()
        icir = ic_mean / ic_std if ic_std != 0 else 0

        # ---- (2) 計算 LS 報酬與 Sharpe ----
        LS_ret = (factor_to_weight(yp[target.notna()], True) * target).sum(axis=1)
        if LS_ret.notna().sum() / len(LS_ret) < 0.9:
            return -10, None

        if LS_ret.std() == 0:
            return -10, None

        mega_ret = pd.concat([scope['ts_result'], LS_ret], axis=1).mean(axis=1)
        sharpe = mega_ret.mean() / mega_ret.std() * 252 ** 0.5

        # ---- (3) 綜合評分公式 ----
        score = (
            10 * ic_mean +
            1 * icir +
            0.2 * sharpe -
            scope['mega_score'] -
            total_length / 1000
        )

        return score, LS_ret

    except Exception as e:
        raise ValueError(f'{原始_expr} 表达式运算失败: {e}')

def preprocessing(individual, toolbox_compile, target, time_constraint_from, time_constraint_to, scope, require_cols):
    
    print(individual)
    #DB_Handler = scope['DB_Handler']
    #func = toolbox_compile(expr=individual)
    #CacheHandler.func_dict = {**CacheHandler.func_dict,**operators_v4.Alpha_F}
    print('准备完成')
    #print('func({})'.format(', '.join(require_cols)))
    #yp = eval('func({})'.format(', '.join(require_cols)), CacheHandler)
    try:
        #yp = eval(f'fast_Neutralization_v2({str(individual)},mega_factor)',CacheHandler)
        yp = eval(str(individual),CacheHandler)
    except:
        yp = pd.DataFrame(np.nan,index = target.index,columns=target.columns)
        #raise ValueError(f'{individual} 表达式运算失败')
    print('运算完成')
    yp = yp[(yp.index >= time_constraint_from)
            & (yp.index <= time_constraint_to)]
    target = target[(target.index >= time_constraint_from)
                         & (target.index <= time_constraint_to)]

    #持倉塞選
    punish = 0
    factor_trade = yp.notna().sum(axis=1)
    average_tradeing_days_ratio = factor_trade.astype(bool).mean()
    average_number_of_trading_companies = factor_trade.mean()
    
    if average_tradeing_days_ratio < 0.9:
        punish-=1000000
    if average_number_of_trading_companies < 100:
        punish-=1000000
    print('前处理完成')
    return {'yp':yp,'target':target,'punish':punish}

def parse_expression(expr,only_get_length = True):
    functions = list()
    data = list()
    params = list()
    def dfs(sub_expr):
        if '(' in sub_expr: 
            # Extract the function name
            function_name = re.search(r'(\w+)\(', sub_expr).group(1)
            functions.append(function_name)
            
            # Remove function name and outer parenthesis
            sub_expr = sub_expr[len(function_name)+1:-1]
            
            # Recursively parse arguments
            depth = 0
            arg_start = 0
            for i, char in enumerate(sub_expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == ',' and depth == 0:
                    dfs(sub_expr[arg_start:i].strip())
                    arg_start = i+1
            dfs(sub_expr[arg_start:].strip())
        else:
            if re.match(r'^\d+$', sub_expr):  # Check if sub_expr is a number
                params.append(sub_expr)
            else:
                data.append(sub_expr)
    dfs(expr)
    total_length = len(functions+data)
    if only_get_length:
        return total_length
    return functions,data,params,total_length