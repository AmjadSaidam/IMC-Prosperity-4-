import pandas as pd
import numpy as np 
from collections import defaultdict 

def missing_count(data: pd.DataFrame): 
    """
    count nas 
    """
    dat: defaultdict = defaultdict()
    for col in data.columns:
        dat[col] = int(data[col].isna().sum())

    return dat

def read_data(paths: list, book = True) -> dict[str, list]: 
    """
    combine books for seperate products
    """
    books: dict = {}
    keys = None
    
    agg_by = 'symbol'
    if book: 
        agg_by = 'product'

    for day, path in enumerate(paths):
        # dataframe
        dat: pd.DataFrame = pd.read_csv(path, sep = ';')
        dat.loc[:, 'day'] = day
        # unique products
        if not books:
            keys = dat[agg_by].unique()
        # for each product construct the data frame
        for k in keys:
            last_book = books[k] if k in books else None
            current_data = dat.loc[dat[agg_by] == k]
            books[k] = pd.concat([last_book, current_data], axis = 0)
    
    return books

def standerdise_timestamp(data): 
    """
    combine timestamp inplace 
    """
    n = data.shape[0]
    data.loc[:, 'timestamp'] = [i for i in range(0, 100*n, 100)]

def clean_book(data): 
    """
    clean data inplace 
    """
    data.replace({'mid_price': 0.0}, np.nan, inplace = True)
    data['mid_price'] = data['mid_price'].ffill()

def total_vols(data: pd.DataFrame):
    """
    aggragates level_2 volume data in level_2 book 
    """
    data.fillna(0, inplace = True)
    bid_vols = ['bid_volume_1', 'bid_volume_2', 'bid_volume_3']
    ask_vols = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3']
    data = data.assign(
        total_bid_volume = data.loc[:, bid_vols].sum(axis = 1), 
        total_ask_volume = data.loc[:, ask_vols].sum(axis = 1)
    )
    return data 

def mid_returns(data: pd.DataFrame, key = 'mid_price'):
    """
    log returns of key 
    """
    data = data.assign(mid_returns = data[key].pct_change().fillna(0))
    return data 

def data_pre_process_pipeline(paths: list[str], book = True):
    """
    data cleaning pipeline
    Book data: reads data, categorsieds by product, cleans book, adds totoal volume column and calculates mid price returns
    """
    df_clean: dict = {}
    df = read_data(paths, book)
    for k in df.keys():
        standerdise_timestamp(df[k]) # inplace 
        if book: 
            clean_book(df[k])
            df[k] = total_vols(df[k])
            df[k] = mid_returns(df[k])
        df_clean[k] = df[k]
    
    return df_clean