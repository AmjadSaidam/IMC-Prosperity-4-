import pandas as pd

def plot_features(x: pd.DataFrame, 
              feature: str, 
              ax):
    """
    plot feature from data, x
    """
    # filter data by day
    xday1 = x.loc[x.day == 0]
    xday2 = x.loc[x.day == 1]
    xday3 = x.loc[x.day == 2]

    # plot 
    ax.plot(xday1.timestamp, getattr(xday1, feature))
    ax.plot(xday2.timestamp, getattr(xday2, feature))
    ax.plot(xday3.timestamp, getattr(xday3, feature))