# Author: Michael Djaballah
# Last edited  PM July 9, 2020
# Last edited by: Michael Djaballah

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from time import sleep, time
from dateutil.relativedelta import relativedelta
import sklearn as sk
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from copy import deepcopy

# Takes no input
# Output is newly saved CSV's containing the current makeup of the S&P 500 
# and its historical additions and removals
# data_path is changeable depending on desired save location
def get_snp_store(data_path='data/'):
    curr_raw = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    
    curr = curr_raw[0]
    hist = curr_raw[1]
    
    new_hist = pd.DataFrame(hist['Date'])
    new_hist['Added'] = hist['Added', 'Ticker']
    new_hist['Removed'] = hist['Removed', 'Ticker']
    
    os.makedirs(data_path, exist_ok=True)
    
    curr.to_csv(data_path + 'snp_current.csv', index=False)
    new_hist.to_csv(data_path + 'snp_hist.csv', index=False)
    return None


# Input: a date in string form with its corresponding format:
# Ex: 'January 1, 2020', '%B %d, %Y'
# Output: a list containing the S&P 500 at the input date
def build_snp(date, date_format='%Y-%m-%d', data_path='data/'):
#     get_snp_store(data_path=data_path)
    
    date_format2 = '%B %d, %Y'
    
    curr = pd.read_csv(data_path + 'snp_current.csv')
    hist = pd.read_csv(data_path + 'snp_hist.csv')
    
    start_date = datetime.strptime(date, date_format)
    
    snp_set = set(curr['Symbol'])
    
    for i in range(len(hist)):
        temp_date = datetime.strptime(hist.iloc[i]['Date'], date_format2)
        if temp_date < start_date:
            break

        tb_removed = hist.iloc[i]['Added']
        tb_added = hist.iloc[i]['Removed']

        if tb_removed in snp_set:
            snp_set.remove(tb_removed)
        if not type(tb_added) == float:
            snp_set.add(tb_added)
    
    return list(snp_set)


# Functions to obtain data from yfinance
# Author Michael Djaballah
# Time last edited: 01:47 AM June 21, 2020
# Last edited by: Michael Djaballah

# Takes a list of tickers as strings
# Output is newly saved CSV's with one CSV per company with daily data
# Saved in specific directory
# data_path is changeable depending on desired save location
def get_data(tickers, start='2000-01-01', data_path='data/', get_new=False):
    interval = '1d'
    prefix = interval + '/'
    
    os.makedirs(data_path + prefix, exist_ok=True)
    
    if get_new:
        curr_tickers = set()
    else:
        curr_tickers = set(os.listdir(data_path + prefix))
    
    for ticker in tickers:
        ticker_label = ticker + '.csv'
        
        if ticker_label not in curr_tickers:
            temp_ticker = yf.Ticker(ticker)
            temp_hist = temp_ticker.history(start=start, interval=interval)
            temp_hist.reset_index(inplace=True)
            temp_hist.dropna(axis=0, inplace=True)
            temp_hist.to_csv(data_path + prefix + ticker_label, index=False)
            sleep(.5)
            
    return None


# Input: Ticker 
# Output: Takes daily data already downloaded and extracts monthly data
# To be used with a set of other functions for intervals named "build_" + interval
# 
def build_1mo(ticker, data_path='data/'):
    if '.csv' in ticker:
        ticker_df = pd.read_csv(data_path + '1d/' + ticker)
    else:
        ticker_df = pd.read_csv(data_path + '1d/' + ticker + '.csv')
        
    date_format = '%Y-%m-%d'
    
    # These are hardcoded and should be more changed if more flexibility is desired
    data_start = '2000-01-01'
    data_end = '2020-05-01'
    
    month_list = list(pd.date_range(data_start, data_end, freq='MS').strftime(date_format))
    
    # Not currently coded for Dividends or Stock Splits
    months_dict = {
        'Date' : [], 
        'Open' : [], 
        'High' : [], 
        'Low' : [], 
        'Close' : [], 
        'Volume' : []
    }
    
    for start in month_list:
        end = datetime.strptime(start, date_format) + relativedelta(months=1) - relativedelta(days=1)
        end = datetime.strftime(end, date_format)
        
        month_df = ticker_df.set_index('Date')[start:end].reset_index()
        
        if len(month_df) > 0:
            months_dict['Date'].append(start)
            months_dict['Open'].append(month_df.iloc[0]['Open'])
            months_dict['High'].append(max(month_df['High']))
            months_dict['Low'].append(min(month_df['Low']))
            months_dict['Close'].append(month_df.iloc[-1]['Close'])
            months_dict['Volume'].append(sum(month_df['Volume']))
    
    months_df = pd.DataFrame.from_dict(months_dict)
    return months_df


# Input: Interval (string) that is in the list of approved
# Output: Data extracted from previously downloaded daily data in 
# a similarly named directory
# The commented out interval set is to be developed 
def build_data(interval, data_path='data/'):
#     interval_set = {'1mo', '5d', '1wk', '3mo'}
    interval_set = {'1mo'}
    if interval not in interval_set:
        print('Invalid interval')
        return -1
    
    prefix = interval + '/'
    
    os.makedirs(data_path + prefix, exist_ok=True)
    
    ticker_labels = os.listdir(data_path + '1d/')
    
    interval_function = globals()['build_' + interval]
    
    for ticker_label in ticker_labels:
        ticker_df = interval_function(ticker_label)
        ticker_df.to_csv(data_path + prefix + ticker_label, index=False)
    return None



# Functions to manipulate and extract desired data from data saved with "get_data"
# Author Michael Djaballah
# Time last edited: 01:47 AM June 21, 2020
# Last edited by: Michael Djaballah


# Takes a ticker as a string
# Output is either a dataframe with desired data, or False, indicating that there was not enough data to build with the desired offset
# data_path is changeable depending on desired save location
def check_ticker(ticker, offset, interval = '1mo', data_path='data/'):
    prefix = interval + '/'
    ticker_df = pd.read_csv(data_path + prefix + ticker + '.csv')
    if len(ticker_df) >= offset:
        return ticker_df
    return False


# Takes a list of tickers as strings, the test depth and historical depth
# Output is a dictionary of data frames: key = ticker string, value = dataframe
# data_path is changeable depending on desired save location
def build_portfolio(tickers, hist_depth, train_depth, start='2000-01-01', interval='1mo', data_path='data/', offset=True, get_new=False):
    if type(offset) == bool:
        offset = train_depth + hist_depth + 60 + 6
        
    prefix = interval + '/'
    
    get_data(tickers, data_path=data_path, start=start, get_new=get_new)
    
    ticker_dict = {}
    
    for ticker in tickers:
        ticker_df = check_ticker(ticker, offset, data_path=data_path)
        if type(ticker_df) != bool:
            ticker_dict[ticker] = ticker_df
    
    return ticker_dict


# Takes a portfolio (from 'build_portfolio'), a desired ticker, date desired to predict on, depth desired, and features desired
# Output is a dataframe with one row or the desired features from previous dates
# Can change 'keep_pred' to True if training or False if predicting
# Target value is present in 'Target' column if 'keep_pred' = True
def build_feature_vector(portfolio, ticker, features, date, hist_depth, target='Close', keep_pred=True):
    ticker_df = portfolio[ticker]
    
    start_date_dt = datetime.strptime(date, '%Y-%m-%d') - relativedelta(months=hist_depth)
    start_date = start_date_dt.strftime('%Y-%m-%d')
    
    feature_df = ticker_df.set_index('Date')[start_date:date].reset_index(drop=True)[features]
    
    new_df_dict = {}
    
    for i in range(len(feature_df)):
        for col in feature_df.columns:
            if i < len(feature_df) - 1:
                new_df_dict[col + ' ' + str(i + 1)] = [feature_df[col].iloc[i]]
            elif col == target:
                if keep_pred:
                    new_df_dict['Target'] = [feature_df[col].iloc[i]]
                    
    new_df = pd.DataFrame.from_dict(new_df_dict)
    
    if keep_pred:
        new_df = new_df[[col for col in new_df.columns if col not in {'Target'}] + ['Target']]
    
    return new_df



# 
# Author Michael Djaballah
# Time last edited: 01:15 AM July 6, 2020
# Last edited by: Michael Djaballah

# Input: portfolio, features, date that the training is to go to, both depths
# Output: dataframe with all features for all training months for the portfolio
# 
# 
def build_train_df(portfolio, features, date, hist_depth, train_depth, target='Close'):    
    tickers = portfolio.keys()
    
    vector_list = []
    for ticker in tickers:
        for i in range(train_depth):
            train_start_dt = datetime.strptime(date, '%Y-%m-%d') - relativedelta(months=(1+i))
            train_start = train_start_dt.strftime('%Y-%m-%d')
            vector_list.append(build_feature_vector(portfolio, ticker, features, train_start, hist_depth, target=target))
            
    feature_df = pd.concat(vector_list)
    return feature_df.reset_index(drop=True)


# Input: portfolio, features, date, historical depth
# Output: dataframe missing y's to predict with and an index list to show which ticker is where in the dataframe
# Returns in a tuple that must be unpacked
# 
def build_test_df(portfolio, features, date, hist_depth, target='Close'):
    tickers = portfolio.keys()
    
    vector_list = []
    index_list = []
    for ticker in tickers:
        vector_list.append(build_feature_vector(portfolio, ticker, features, date, hist_depth, target=target, keep_pred=False))
        index_list.append(ticker)
        
    test_df = pd.concat(vector_list)
    return test_df.reset_index(drop=True), index_list


# Input: portfolio, desired list of tickers to get returns for, and the date of those returns
# Output: list of returns for those tickers
# 
# 
def build_returns(portfolio, tickers, date):
    returns = []
    for ticker in tickers:
        temp_ticker_dict = portfolio[ticker].set_index('Date').loc[date]
        returns.append((temp_ticker_dict['Close'] - temp_ticker_dict['Open'])/temp_ticker_dict['Open'])
    return returns


# Input: Portfolio, ticker, and desired date 
# Output: Whether or not the data has the specific date
# This is to tell if a certain ticker exists on a certain date
# 
def check_date(portfolio, ticker, date):
    dates = set(portfolio[ticker]['Date'])
    return date in dates


# Input: Portfolio, date
# Output: None, this directly edits the portfolio
# This can be changed to reduce side effects, but it is used 
# to remove tickers that are no longer needed moving forward in a backtest
def clean_portfolio(portfolio, date):
    tickers = list(portfolio.keys())
    bad_tickers = []
    for ticker in tickers:
        if not check_date(portfolio, ticker, date):
            bad_tickers.append(ticker)
    for bad_ticker in bad_tickers:
        del portfolio[bad_ticker]
    return None


# ALERT: This function has been depracated as direct access to the scaler is needed for interpretation
# Input: Dataframe
# Output: Dataframe scaled to standard
# 
# 
# def build_scaled_df(dataframe):
#     scaler = StandardScaler()
#     scaled_array = scaler.fit_transform(dataframe)
#     scaled_dataframe = pd.DataFrame(scaled_array, columns=dataframe.columns)
#     return scaled_dataframe


# Input: Portfolio, date
# Output: Dataframe of 'Close 1', the close of the previous month for the whole portfolio
# 
# 
def build_previous_close(portfolio, date):
    tickers = portfolio.keys()
    
    vector_list = []
    index_list = []
    for ticker in tickers:
        vector_list.append(build_feature_vector(portfolio, ticker, ['Close'], date, hist_depth=1, keep_pred=False))
        index_list.append(ticker)
        
    df = pd.concat(vector_list)
    return df.reset_index(drop=True)


# Input: Portfolio, model, date, features, and depths
# Output: 
# 
# 
def build_machine(portfolio, features, model, date, hist_depth, train_depth, use_scaling=True, target='Close'):
    train_df = build_train_df(portfolio, features, date, hist_depth, train_depth, target=target)
    
    X = train_df.loc[:, train_df.columns != 'Target'].values
    y = train_df['Target'].values
    
    model.fit(X, y)
    
    test_df, tickers = build_test_df(portfolio, features, date, hist_depth, target=target)
    X_test = test_df.values
    
    predicted_returns = model.predict(X_test)
    
    previous_close = build_previous_close(portfolio, date)['Close 1'].values
        
    if use_scaling:
        predicted_returns = (predicted_returns - previous_close)/previous_close
    
    predicted_returns = list(predicted_returns)
    
    returns_dict = {}
    
    for i in range(len(tickers)):
        returns_dict[tickers[i]] = predicted_returns[i]
    
    return returns_dict


# Input: Returns dictionary and fixed long short parameters
# Output: Allocation dictionary for long and short tickers
# 
# 
def fixed_long_short(returns_dict, long=15, short=0):
    allocation = {
        'long' : [], 
        'short' : []
    }
    top = sorted(returns_dict.items(), key=lambda x: x[1])[::-1]
    sorted_tickers = [x[0] for x in top]
    
    allocation['long'] = sorted_tickers[:long]
    if short == 0:
        allocation['short'] = []
    else:
        allocation['short'] = sorted_tickers[-1 * short:]
#     print('long: ')
#     print(allocation['long'])
#     print('short: ')
#     print(allocation['short'])
    
    return allocation
    

# Input: all things needed to build a machine with a date range to test
# Output: per date average returns as determined by the model
# 
# 
def backtest(portfolio, features, model, hist_depth, train_depth, start_date, end_date, 
             allocation_builder=fixed_long_short, params={}, blacklist=set(), 
             target='Close', use_scaling=True):
    
    months = list(pd.date_range(start_date, end_date, freq='MS').strftime('%Y-%m-%d'))
    
    for ticker in blacklist:
        if ticker in portfolio:
            del portfolio[ticker]
            
    portfolio = deepcopy(portfolio)
    
    overall_returns = []
    specific_returns = []
    for month in months:
        start_time = time()
        
        clean_portfolio(portfolio, month)
        
        returns_dict = build_machine(portfolio, features, model, month, hist_depth, train_depth, target=target, use_scaling=use_scaling)
    
        allocation = allocation_builder(returns_dict, **params)
        
        long_returns = build_returns(portfolio, allocation['long'], month)
        short_returns = build_returns(portfolio, allocation['short'], month)
        short_returns = [short_return * -1 for short_return in short_returns]
        total_returns = long_returns + short_returns
        average_returns = sum(total_returns)/len(total_returns)
        
        
        specific_returns_dict = {'long': {}, 'short': {}}
        for i in range(len(allocation['long'])):
            specific_returns_dict['long'][allocation['long'][i]] = long_returns[i]

        for i in range(len(allocation['short'])):
            specific_returns_dict['short'][allocation['short'][i]] = short_returns[i]
            
        specific_returns.append(specific_returns_dict)
        overall_returns.append(average_returns)
        
        print(month, round(average_returns, 5), round(time() - start_time, 2))
        
    build_documentation(
        portfolio, 
        features, 
        model, 
        hist_depth, 
        train_depth, 
        start_date, 
        end_date, 
        allocation_builder=allocation_builder, 
        params=params, 
        blacklist=blacklist, 
        target=target, 
        returns=overall_returns, 
        specific_ret=specific_returns
    )
    
    return overall_returns, specific_returns



# METRICS
# Author Michael Djaballah
# Time last edited: 05:31 PM July 12, 2020
# Last edited by: Michael Djaballah

# Sharpe Ratio
# Total Return DONE
# Volatility DONE
# Maximum Drawdown DONE 
# Lowest balance DONE
# Annualized Returns DONE

def cumulative_returns(returns):
    starting_capital = 1
    historical_returns = [starting_capital]
    for i in range(len(returns)):
        starting_capital *= (returns[i] + 1)
        historical_returns.append(starting_capital)
    return historical_returns


def total_return(cumulative_returns):
    return cumulative_returns[-1]


def return_volatility(returns):
    return np.std(returns)
    

def max_drawdown(cumulative_returns):
    drawdowns = []
    for i in range(len(cumulative_returns)):
        for j in range(i+1, len(cumulative_returns)):
            drawdowns.append((cumulative_returns[i] - cumulative_returns[j])/cumulative_returns[i])
    return max(drawdowns)


def min_amount(cumulative_returns):
    return min(cumulative_returns)


def sharpe_ratio(returns):
    return np.sqrt(12) * np.mean(returns)/np.std(returns)


def annualized_returns(returns,  unit=1/12, tot_return=None):
    if type(tot_return) != float:
        cumul_returns = cumulative_returns(returns)
        tot_return = total_return(cumul_returns)
    years = len(returns) * unit
    return tot_return**(1/years)


def universe_backtest(portfolio, start_date, end_date):
    months = list(pd.date_range(start_date, end_date, freq='MS').strftime('%Y-%m-%d'))
    
    returns = []
    
    portfolio = deepcopy(portfolio)
    
    for month in months:
        clean_portfolio(portfolio, month)
        monthly_returns = build_returns(portfolio, list(portfolio.keys()), month)
        returns.append(sum(monthly_returns)/len(monthly_returns))
    return returns


# 
# 
# 
# 

def build_documentation(portfolio, features, model, hist_depth, train_depth, start_date, end_date, 
             allocation_builder, params, blacklist, returns, specific_ret, target='Close'):
    path = 'results/'
    tickers = list(portfolio.keys())
    
    cumul_returns = cumulative_returns(returns)
    return_vol = return_volatility(returns)
    total_returns = total_return(cumul_returns)
    ann_returns = annualized_returns(returns, tot_return=total_returns)
    max_draw = max_drawdown(cumul_returns)
    min_balance = min_amount(cumul_returns)
    sharpe = sharpe_ratio(returns)
    
    docu_dict = {
        'tickers' : [tickers], 
        'features' : [features], 
        'model' : [str(model)], 
        'hist_depth' : [hist_depth], 
        'train_depth' : [train_depth], 
        'start_date' : [start_date], 
        'end_date' : [end_date], 
        'allocation_builder' : [str(allocation_builder)], 
        'params' : [params], 
        'blacklist' : [blacklist], 
        'target' : [target], 
        'returns' : [returns], 
        'cumulative returns' : [cumul_returns], 
        'total return' : [total_returns], 
        'sharpe ratio: ' : [sharpe], 
        'annualized return' : [ann_returns],
        'return volatility' : [return_vol], 
        'max drawdown' : [max_draw], 
        'minimum' : [min_balance]
    }
    
    docu_df = pd.DataFrame.from_dict(docu_dict)
    docu_df = docu_df.T
    docu_df.columns = ['Portfolio']
    
    os.makedirs(path, exist_ok=True)
    
    if 'snp.csv' not in set(os.listdir(path)):
        snp_returns = universe_backtest(tickers, start_date=start_date, end_date=end)
        
        snp_cumul = cumul_returns(snp_returns)
        snp_vol = return_volatility(snp_returns)
        snp_total = total_return(snp_cumul)
        snp_ann = annualized_returns(snp_returns, tot_return=snp_total)
        snp_max_draw = max_drawdown(snp_cumul)
        snp_min_balance = min_amount(snp_cumul)
        snp_sharpe = sharpe_ratio(snp_returns)

        snp_dict = {
            'tickers' : ['N/A'], 
            'features' : ['N/A'], 
            'model' : ['N/A'], 
            'hist_depth' : ['N/A'], 
            'train_depth' : ['N/A'], 
            'start_date' : [start_date], 
            'end_date' : [end_date], 
            'allocation_builder' : ['N/A'], 
            'params' : ['N/A'], 
            'blacklist' : [blacklist], 
            'target' : ['N/A'], 
            'returns' : [snp_returns], 
            'cumulative returns' : [snp_cumul], 
            'total return' : [snp_total], 
            'sharpe ratio' : [snp_sharpe], 
            'annualized return' : [snp_ann],
            'return volatility' : [snp_vol], 
            'max drawdown' : [snp_max_draw], 
            'minimum' : [snp_min_balance]
        }
        
        snp_df = pd.DataFrame.from_dict(snp_dict)
        snp_df = snp_df.T
        snp_df.columns = ['S&P']
        
        snp_df.to_csv(path + 'snp.csv')
    
    snp_df = pd.read_csv(path + 'snp.csv', index_col=0)
    
    tot_df = docu_df.join(snp_df)
    
    print(tot_df)
    
    label = str(datetime.now())
    
    tot_df.to_csv(path + label + '.csv')
    
    spec_df = specific_returns_df(specific_ret, start_date, end_date)
    spec_df.to_csv(path + 'specific_' + label + '.csv')
    
    snp_cumul = [float(x) for x in snp_df.loc['cumulative returns'][0].replace('[', '').replace(']', '').split(', ')]
    
    # 
    df_dict = {'S&P': snp_cumul, 'Port': cumul_returns}
    compare_df = pd.DataFrame.from_dict(df_dict)

    fig_dims = (20,10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.lineplot(data=compare_df)
    plt.show()
    
    print()
    print('Specifics on the backtest can be found at: ' + path + 'specific_' + label + '.csv')
    
    
def specific_returns_df(specific_results, start_date, end_date):
    months = list(pd.date_range(start_date, end_date, freq='MS').strftime('%Y-%m-%d'))

    specific_dict = {}
    for i in range(len(months)):
        specific_dict[months[i]] = [specific_results[i]['long'], specific_results[i]['short']]

    spec_df = pd.DataFrame.from_dict(specific_dict)
    spec_df = spec_df.T
    spec_df.columns = ['long', 'short']
    return spec_df


def build_inputs():
    # Historical depth, default to 24
    # Training depth, default to 3
    # Features, default to Close, Volume
    # Start date, default to 2015-01-01
    # End Date default to 2019-12-01
    # Allocation builder, default to fixed long short
    # Allocation parameters, default to long : 15, short : 0
    # Model to be used, default to random forest regressor
    inputs = {
        'hist_depth' : 24,
        'train_depth' : 3, 
        'features' : [],
        'start_date' : '2015-01-01',
        'end_date' : '2019-12-01', 
        'allocation_builder' : fixed_long_short,
        'params' : {
            'long': 15,
            'short' : 0
        }, 
        'model' : RandomForestRegressor(random_state=407, n_jobs=-1)
    }
    
    hist_depth_bool = True
    train_depth_bool = True
    features_bool = True
    start_date_bool = True
    end_date_bool = True
    allocation_parameters_bool = True
    model_bool = True
    
    hist_temp = input('Enter a historical depth or press enter to default to 24: ')
    if hist_temp != '':
        inputs['hist_depth'] = int(hist_temp)
        
    train_temp = input('Enter a training depth or press enter to default to 3: ')
    if train_temp != '':
        inputs['train_depth'] = int(train_temp)
    
    features_temp = input('Enter features to be used separated by a space or press enter for Close and Volume: ')
    if features_temp != '':
        inputs['features'] = features_temp.split(' ')
    else:
        inputs['features'] = ['Close', 'Volume']
    
    start_temp = input('Enter a start date in YYYY-MM-DD (this month will be included) or press enter for "2015-01-01": ')
    if start_temp != '':
        inputs['start_date'] = start_temp
    
    end_temp = input('Enter an end date in YYYY-MM-DD (this month will be included) or press enter for "2019-12-01": ')
    if end_temp != '':
        inputs['end_date'] = end_temp
        
    print('Allocation builder defaults to fixed_long_short')
    param_temp = input('Enter the desired long and short stock amounts separated by a space or press enter for 15 long and 0 short: ')
    if param_temp != '':
        param_list = param_temp.split(' ')
        inputs['params']['long'] = int(param_list[0])
        inputs['params']['short'] = int(param_list[1])
    
    model_temp = input('Would you like to use a gradient boosted algorithm? Enter "Y" or hit enter to use a random forest: ')
    if model_temp == 'y':
        inputs['model'] = GradientBoostingRegressor(random_state=407)
        
    return inputs


def main():
    inputs = build_inputs()
    
    universe = build_snp('2015-01-01')
    
    port = build_portfolio(universe, inputs['hist_depth'], inputs['train_depth'])
    
    backtest(portfolio=port, **inputs)
    
    
if __name__ == '__main__':
    main()