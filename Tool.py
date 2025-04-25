import os
import numpy as np
import pandas as pd
import operators_v4
import tqdm
import quantstats as qs
qs.extend_pandas()
import cufflinks as cf
cf.go_offline()
class Handler(dict):
    def __init__(self, path, data_type: str = 'parquet'):
        self.path = path
        self.cashe_dict = {}
        self.func_dict = operators_v4.Alpha_F
        if data_type == 'pickle':
            data_type = 'pkl'
        self.data_type = data_type
        self.reindex_like = None

        os.makedirs(path, exist_ok=True)

    def __getitem__(self, key):
        if key in self.cashe_dict:
            return self.cashe_dict[key]
        elif key in self.func_dict:
            return self.func_dict[key]
        else:
            file_path = os.path.join(self.path, f'{key}.{self.data_type}')
            if os.path.exists(file_path):
                try:
                    if self.data_type == 'parquet':
                        data = pd.read_parquet(file_path)
                    elif self.data_type == 'pkl':
                        data = pd.read_pickle(file_path)
                    else:
                        raise ValueError(f'不支援的格式: {self.data_type}')

                    if self.reindex_like is not None:
                        data = data.reindex_like(self.reindex_like)

                    data = data.astype(float, errors='ignore')
                    self.cashe_dict[key] = data
                    return data
                except Exception as e:
                    raise ValueError(f'讀取失敗: {file_path}，錯誤: {e}')
            raise ValueError(f'找不到資料檔案：{file_path}')

    def __call__(self, key):
        return self.__getitem__(key)

    def __setitem__(self, key, value):
        file_path = os.path.join(self.path, f'{key}.{self.data_type}')

        # 處理 DataFrame
        if isinstance(value, pd.DataFrame):
            value = value.copy()
            numeric_cols = value.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                value[col] = value[col].where(np.isfinite(value[col]))
            # 非數值欄位保留不動
        elif isinstance(value, pd.Series):
            if np.issubdtype(value.dtype, np.number):
                value = value.where(np.isfinite(value))
            else:
                pass  # 不處理非數值欄位

        else:
            raise TypeError(f"{key} 的資料型態 {type(value)} 無法被儲存")

        # 儲存檔案
        if self.data_type == 'parquet':
            value.to_parquet(file_path)
        elif self.data_type == 'pkl':
            value.to_pickle(file_path)
        else:
            raise ValueError(f"不支援的格式：{self.data_type}")

        self.cashe_dict[key] = value

    def cash_list(self):
        return sorted([
            filename[:-(len(self.data_type) + 1)]
            for filename in os.listdir(self.path)
            if filename.endswith(f".{self.data_type}")
        ])

def max_drawdown(prices):
    # 計算累計的最大值
    cumulative_max = prices.cummax()
    # 計算回撤 (Drawdown)
    drawdown = (prices - cumulative_max) / cumulative_max
    # 計算最大回撤 (MDD)
    mdd = drawdown.min()
    return mdd
def show_stats(bt_ret:pd.DataFrame)->None:
    if isinstance(bt_ret,pd.Series):
        bt_ret = pd.DataFrame({'策略':bt_ret})
    try:
        display(pd.concat({"CAGR(%)":bt_ret.cagr()*100,
                'Sharpe':bt_ret.mean()/bt_ret.std()*252**0.5,
                'Calmar':bt_ret.calmar(),
                'MDD(%)':bt_ret.max_drawdown()*100,
                '單利MDD(%)' : max_drawdown(bt_ret.cumsum().add(1))*100,
                '样本胜率(%)':bt_ret.apply(lambda X:((X.dropna()>0).sum()  / X.dropna().shape[0])*100),
                '周胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('W').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('W').prod().sub(1).dropna().shape[0])*100),
                '月胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('ME').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('ME').prod().sub(1).shape[0])*100),
                '年胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('YE').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('YE').prod().sub(1).shape[0])*100),
                '盈亏比(avg_win/avg_loss)': bt_ret.apply(lambda X:(X[X > 0].mean() / abs(X[X < 0].mean()))),
                '总赚赔比(profit_factor)':bt_ret.profit_factor(),
                '预期报酬(bps)':((1 + bt_ret).prod() ** (1 / len(bt_ret)) - 1)*10000,
                '样本数':bt_ret.apply(lambda X:X.dropna().count()),
                },axis = 1).round(2))
    except:
        display(pd.concat({"CAGR(%)":bt_ret.cagr()*100,
                'Sharpe':bt_ret.mean()/bt_ret.std()*252**0.5,
                'MDD(%)':bt_ret.max_drawdown()*100,
                '單利MDD(%)' : max_drawdown(bt_ret.cumsum().add(1))*100,
                '样本胜率(%)':bt_ret.apply(lambda X:((X.dropna()>0).sum()  / X.dropna().shape[0])*100),
                '周胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('W').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('W').prod().sub(1).dropna().shape[0])*100),
                '月胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('M').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('M').prod().sub(1).shape[0])*100),
                '年胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('Y').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('Y').prod().sub(1).shape[0])*100),
                '盈亏比(avg_win/avg_loss)': bt_ret.apply(lambda X:(X[X > 0].mean() / abs(X[X < 0].mean()))),
                '总赚赔比(profit_factor)':bt_ret.profit_factor(),
                '预期报酬(bps)':((1 + bt_ret).prod() ** (1 / len(bt_ret)) - 1)*10000,
                '样本数':bt_ret.apply(lambda X:X.dropna().count()),
                },axis = 1).round(2))
def backtest_factor(factor:pd.DataFrame,exp_ret:pd.DataFrame,rank_range_n:int = 10,start_date:str = '2019-01-01'):
    factor_rank = factor.rank(axis = 1,pct = True,method = 'first')

    IC_Se = factor.corrwith(exp_ret,axis=1,method='spearman').sort_index().loc[start_date:]
    print(f'IC_mean:{round(IC_Se.mean(),4)}')
    print(f'IC_IR:{round(IC_Se.mean()/IC_Se.std(),4)}')

    bt = pd.concat({f'{int(((_/rank_range_n)*100))}% ~ {int((_+1)/rank_range_n*100)}%':exp_ret[(factor_rank>_/rank_range_n) & (factor_rank<=(_+1)/rank_range_n)].mean(axis = 1) - exp_ret.mean(axis=1) for _ in tqdm.tqdm(range(rank_range_n))}, axis = 1).dropna(how = 'all')
    bt = bt.loc[start_date:]
    if (bt.iloc[:,-1] - bt.iloc[:,0]).add(1).prod() > 1:
        bt['LS_ret'] = bt.iloc[:,-1] - bt.iloc[:,0]
    else:
        bt['LS_ret'] = bt.iloc[:,0] - bt.iloc[:,-1]
    show_stats(bt)

    (bt.drop(columns='LS_ret').loc[start_date:].cagr()*100).iplot(kind = 'bar')
    bt.index = bt.index.astype(str)
    bt.cumsum().ffill().iplot()
    bt.index = pd.to_datetime(bt.index)
CAGR = lambda bt_ret: (abs(bt_ret.add(1).prod()) ** (1 / (len(bt_ret) / 12)) - 1) * 100