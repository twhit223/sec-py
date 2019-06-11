import re
import os
from time import gmtime, strftime
from datetime import datetime, timedelta
import unicodedata
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import bs4 as bs
from lxml import html
from tqdm import tqdm
import code
from definitions import DATA_DIR
import pickle

'''
GetTickers() gets all tickers for stocks currently listed on the NASDAQ, NYSE, and AMEX and stores them in a list. 
parameters: N/A
return: tickers - a list of tickers 
'''
def GetTickers():
  
  # Check to see if the tickers have already been downloaded. If there is a file with the data already saved, load it. Otherwise tokenize and save the data.
  for file in os.listdir(DATA_DIR + '/tickers'):
    if file == ('tickers.txt'):
      print("Loading tickers...")
      with open(DATA_DIR + '/tickers/tickers.txt', 'rb') as f:
        tickers = pickle.load(f)
      return tickers

  # Get lists of tickers from NASDAQ, NYSE, AMEX
  print("Fetching tickers...")
  nasdaq_tickers = pd.read_csv('https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download')
  nyse_tickers = pd.read_csv('https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download')
  amex_tickers = pd.read_csv('https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download')

  # Drop irrelevant cols
  nasdaq_tickers.drop(labels='Unnamed: 8', axis='columns', inplace=True)
  nyse_tickers.drop(labels='Unnamed: 8', axis='columns', inplace=True)
  amex_tickers.drop(labels='Unnamed: 8', axis='columns', inplace=True)

  # Create full list of tickers/names across all 3 exchanges
  tickers = list(set(list(nasdaq_tickers['Symbol']) + list(nyse_tickers['Symbol']) + list(amex_tickers['Symbol'])))

  # Save the tickers
  print("Saving tickers...")
  with open(DATA_DIR + '/tickers/tickers.txt', 'wb') as f:
    pickle.dump(tickers, f)

  return tickers

'''
AddSicToCikDict
parameters: cik_dict - a dataframe that contains a row by row mapping from ticker to CIK
return: cik_dict - a dataframe that contains a row by row mapping from ticker to CIK and SIC code
'''
def AddSicToCikDict(ticker_cik_df):
  
  # Load the CIK to SIC mapping downloaded from WRDS
  cik_sic_map = pd.read_excel(DATA_DIR + '/tickers/cik_sic_map.xlsx')
  cik_sic_map = cik_sic_map[['cik', 'sic']].drop_duplicates()
  cik_sic_map['cik'] = cik_sic_map['cik'].apply(str)
  cik_sic_map['cik'] = cik_sic_map['cik'].apply(lambda x: x.zfill(10))


  # Merge the SIC code onto cik_dict
  ticker_cik_df = ticker_cik_df.merge(cik_sic_map, how = 'left', on = 'cik').drop_duplicates()

  # Check for NA's after the merge
  missing_comps = ticker_cik_df.loc[ticker_cik_df['sic'].isna()]
  print("The following {} companies are missing a SIC code: ".format(len(missing_comps)))
  print(missing_comps)


  code.interact(local = locals())

  return ticker_cik_df


'''
MapTickerToCik gets the tickers and maps them to a Central Index Key (CIK). THe SEC indexes company filings using a CIK rather than a stock ticker.
parameters: tickers - a list of stock tickers
return: cik_dict - a dictionary that maps tickers to cik
'''
def MapTickerToCik(tickers):

  # Check to see if the ticker to cik map has already been created. If there is a file with the data already saved, load it. Otherwise create the map. 
  for file in os.listdir(DATA_DIR + '/tickers'):
    if file == ('ticker_cik_df.pickle'):
      print("Loading ticker_cik_df...")
      with open(DATA_DIR + '/tickers/ticker_cik_df.pickle', 'rb') as f:
        ticker_cik_df = pickle.load(f)
      return ticker_cik_df

  url = 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'
  cik_re = re.compile(r'.*CIK=(\d{10}).*')

  cik_dict = {}
  for ticker in tqdm(tickers): # Use tqdm lib for progress bar
      results = cik_re.findall(requests.get(url.format(ticker)).text)
      if len(results):
          cik_dict[str(ticker).lower()] = str(results[0])

  # Clean up the ticker-CIK mapping as a DataFrame
  ticker_cik_df = pd.DataFrame.from_dict(data=cik_dict, orient='index')
  ticker_cik_df.reset_index(inplace=True)
  ticker_cik_df.columns = ['ticker', 'cik']
  ticker_cik_df['cik'] = [str(cik) for cik in ticker_cik_df['cik']]

  # Check for duplicates
  CheckDuplicateTickers(ticker_cik_df)

  # Keep first ticker alphabetically for duplicated CIKs
  ticker_cik_df = ticker_cik_df.sort_values(by='ticker')
  ticker_cik_df.drop_duplicates(subset='cik', keep='first', inplace=True)
 
  # Check for duplicates again
  CheckDuplicateTickers(ticker_cik_df)

  # Add the SIC codes to the cik_dict
  ticker_cik_df = AddSicToCikDict(ticker_cik_df)

  # Save the ticker_cik_df mapping
  print("Saving ticker_cik_df...")
  pickle.dump(ticker_cik_df, open(DATA_DIR + '/tickers/ticker_cik_df.pickle', 'wb'))
  
  return ticker_cik_df


'''
CheckDuplicateTickers is a helper function that prints the number of unique tickers, CIKs, and ticker-CIK parings.
parameters: ticker_cik_df - a dataframe that contains a row by row mapping from ticker to CIK
return: N/A
'''
def CheckDuplicateTickers(ticker_cik_df):
  print("Number of ticker-cik pairings:", len(ticker_cik_df))
  print("Number of unique tickers:", len(set(ticker_cik_df['ticker'])))
  print("Number of unique CIKs:", len(set(ticker_cik_df['cik'])))



tickers = GetTickers()
ticker_cik_df = MapTickerToCik(tickers)
code.interact(local = locals())

