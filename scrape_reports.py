import re
import os
import shutil
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
from definitions import DATA_DIR, PATHNAME_10K, PATHNAME_10Q, PATHNAME_S1
import pickle

def WriteLogFile(log_file_name, text):
  
  '''
  Helper function.
  Writes a log file with all notes and
  error messages from a scraping "session".
  
  Parameters
  ----------
  log_file_name : str
    Name of the log file (should be a .txt file).
  text : str
    Text to write to the log file.
      
  Returns
  -------
  None.
  
  '''
  
  with open(log_file_name, "a") as log_file:
      log_file.write(text)

  return


def GetTickerCikDf():

  ''' 
  Gets the ticker_cik_df created by get_ticker_cik.py
  If none is found, throws an error.

  Parameters
  ----------
  None.

  Returns
  _______
  ticker_cik_df : pandas df
    dataframe containing cik, ticker, and sic code for all public companies
  '''

  with open(DATA_DIR + '/tickers/sic_7370_ticker_cik_df.pickle', 'rb') as f:
    ticker_cik_df = pickle.load(f)

  return ticker_cik_df


def Scrape10K(browse_url_base, filing_url_base, doc_url_base, cik, ticker, log_file_name):
    
  '''
  Scrapes all 10-Ks and 10-K405s for a particular 
  CIK from EDGAR.
  
  Parameters
  ----------
  browse_url_base : str
      Base URL for browsing EDGAR.
  filing_url_base : str
      Base URL for filings listings on EDGAR.
  doc_url_base : str
      Base URL for one filing's document tables
      page on EDGAR.
  cik : str
      Central Index Key.
  log_file_name : str
      Name of the log file (should be a .txt file).
      
  Returns
  -------
  None.
  
  '''
  
  # Check if we've already scraped this CIK
  try:
      directory = str(cik) + "_" + ticker
      os.mkdir(directory)
  except OSError:
      print("Already scraped CIK", cik)
      return
  
  # If we haven't, go into the directory for that CIK
  os.chdir(directory)
  
  print('Scraping CIK', cik)
  
  # Request list of 10-K filings
  res = requests.get(browse_url_base % cik)
  
  # If the request failed, log the failure and exit
  if res.status_code != 200:
      os.chdir('..')
      os.rmdir(directory) # remove empty dir
      text = "Request failed with error code " + str(res.status_code) + \
             "\nFailed URL: " + (browse_url_base % cik) + '\n'
      WriteLogFile(log_file_name, text)
      return

  # If the request doesn't fail, continue...
  
  # Parse the response HTML using BeautifulSoup
  soup = bs.BeautifulSoup(res.text, "lxml")

  # Extract all tables from the response
  html_tables = soup.find_all('table')
  
  # Check that the table we're looking for exists
  # If it doesn't, exit
  if len(html_tables)<3:
      os.chdir('..')
      return
  
  # Parse the Filings table
  filings_table = pd.read_html(str(html_tables[2]), header=0)[0]
  filings_table['Filings'] = [str(x) for x in filings_table['Filings']]

  # Get only 10-K and 10-K405 document filings
  filings_table = filings_table[(filings_table['Filings'] == '10-K') | (filings_table['Filings'] == '10-K405')]

  # If filings table doesn't have any
  # 10-Ks or 10-K405s, exit
  if len(filings_table)==0:
      os.chdir('..')
      return
  
  # Get accession number for each 10-K and 10-K405 filing
  filings_table['Acc_No'] = [x.replace('\xa0',' ')
                             .split('Acc-no: ')[1]
                             .split(' ')[0] for x in filings_table['Description']]

  # Iterate through each filing and 
  # scrape the corresponding document...
  for index, row in filings_table.iterrows():
      
      # Get the accession number for the filing
      acc_no = str(row['Acc_No'])
      
      # Navigate to the page for the filing
      docs_page = requests.get(filing_url_base % (cik, acc_no))
      
      # If request fails, log the failure
      # and skip to the next filing
      if docs_page.status_code != 200:
          os.chdir('..')
          text = "Request failed with error code " + str(docs_page.status_code) + \
                 "\nFailed URL: " + (filing_url_base % (cik, acc_no)) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue

      # If request succeeds, keep going...
      
      # Parse the table of documents for the filing
      docs_page_soup = bs.BeautifulSoup(docs_page.text, 'lxml')
      docs_html_tables = docs_page_soup.find_all('table')
      if len(docs_html_tables)==0:
          continue
      docs_table = pd.read_html(str(docs_html_tables[0]), header=0)[0]
      docs_table['Type'] = [str(x) for x in docs_table['Type']]
      
      # Get the 10-K and 10-K405 entries for the filing
      docs_table = docs_table[(docs_table['Type'] == '10-K') | (docs_table['Type'] == '10-K405')]
      
      # If there aren't any 10-K or 10-K405 entries,
      # skip to the next filing
      if len(docs_table)==0:
          continue
      # If there are 10-K or 10-K405 entries,
      # grab the first document
      elif len(docs_table)>0:
          docs_table = docs_table.iloc[0]
      
      docname = docs_table['Document']
      
      # If that first entry is unavailable,
      # log the failure and exit
      if str(docname) == 'nan':
          os.chdir('..')
          text = 'File with CIK: %s and Acc_No: %s is unavailable' % (cik, acc_no) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue       
      
      # If it is available, continue...
      
      # Request the file
      file = requests.get(doc_url_base % (cik, acc_no.replace('-', ''), docname))
      
      # If the request fails, log the failure and exit
      if file.status_code != 200:
          os.chdir('..')
          text = "Request failed with error code " + str(file.status_code) + \
                 "\nFailed URL: " + (doc_url_base % (cik, acc_no.replace('-', ''), docname)) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue
      
      # If it succeeds, keep going...
      
      # Save the file in appropriate format
      if '.txt' in docname:
          # Save text as TXT
          date = str(row['Filing Date'])
          filename = cik + '_' + date + '.txt'
          html_file = open(filename, 'a')
          html_file.write(file.text)
          html_file.close()
      else:
          # Save text as HTML
          date = str(row['Filing Date'])
          filename = cik + '_' + date + '.html'
          html_file = open(filename, 'a')
          html_file.write(file.text)
          html_file.close()
      
  # Move back to the main 10-K directory
  os.chdir('..')
      
  return


def Scrape10Q(browse_url_base, filing_url_base, doc_url_base, cik, ticker, log_file_name):
  
  '''
  Scrapes all 10-Qs for a particular CIK from EDGAR.
  
  Parameters
  ----------
  browse_url_base : str
      Base URL for browsing EDGAR.
  filing_url_base : str
      Base URL for filings listings on EDGAR.
  doc_url_base : str
      Base URL for one filing's document tables
      page on EDGAR.
  cik : str
      Central Index Key.
  log_file_name : str
      Name of the log file (should be a .txt file).
      
  Returns
  -------
  None.
  
  '''
  
  # Check if we've already scraped this CIK
  try:
      directory = str(cik) + '_' + ticker
      os.mkdir(directory)
  except OSError:
      print("Already scraped CIK", cik)
      return
  
  # If we haven't, go into the directory for that CIK
  os.chdir(directory)
  
  print('Scraping CIK', cik)
  
  # Request list of 10-Q filings
  res = requests.get(browse_url_base % cik)
  
  # If the request failed, log the failure and exit
  if res.status_code != 200:
      os.chdir('..')
      os.rmdir(directory) # remove empty dir
      text = "Request failed with error code " + str(res.status_code) + \
             "\nFailed URL: " + (browse_url_base % cik) + '\n'
      WriteLogFile(log_file_name, text)
      return
  
  # If the request doesn't fail, continue...

  # Parse the response HTML using BeautifulSoup
  soup = bs.BeautifulSoup(res.text, "lxml")

  # Extract all tables from the response
  html_tables = soup.find_all('table')
  
  # Check that the table we're looking for exists
  # If it doesn't, exit
  if len(html_tables)<3:
      print("table too short")
      os.chdir('..')
      return
  
  # Parse the Filings table
  filings_table = pd.read_html(str(html_tables[2]), header=0)[0]
  filings_table['Filings'] = [str(x) for x in filings_table['Filings']]

  # Get only 10-Q document filings
  filings_table = filings_table[filings_table['Filings'] == '10-Q']

  # If filings table doesn't have any
  # 10-Ks or 10-K405s, exit
  if len(filings_table)==0:
      os.chdir('..')
      return
  
  # Get accession number for each 10-K and 10-K405 filing
  filings_table['Acc_No'] = [x.replace('\xa0',' ')
                             .split('Acc-no: ')[1]
                             .split(' ')[0] for x in filings_table['Description']]

  # Iterate through each filing and 
  # scrape the corresponding document...
  for index, row in filings_table.iterrows():
      
      # Get the accession number for the filing
      acc_no = str(row['Acc_No'])
      
      # Navigate to the page for the filing
      docs_page = requests.get(filing_url_base % (cik, acc_no))
      
      # If request fails, log the failure
      # and skip to the next filing    
      if docs_page.status_code != 200:
          os.chdir('..')
          text = "Request failed with error code " + str(docs_page.status_code) + \
                 "\nFailed URL: " + (filing_url_base % (cik, acc_no)) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue
          
      # If request succeeds, keep going...
      
      # Parse the table of documents for the filing
      docs_page_soup = bs.BeautifulSoup(docs_page.text, 'lxml')
      docs_html_tables = docs_page_soup.find_all('table')
      if len(docs_html_tables)==0:
          continue
      docs_table = pd.read_html(str(docs_html_tables[0]), header=0)[0]
      docs_table['Type'] = [str(x) for x in docs_table['Type']]
      
      # Get the 10-K and 10-K405 entries for the filing
      docs_table = docs_table[docs_table['Type'] == '10-Q']
      
      # If there aren't any 10-K or 10-K405 entries,
      # skip to the next filing
      if len(docs_table)==0:
          continue
      # If there are 10-K or 10-K405 entries,
      # grab the first document
      elif len(docs_table)>0:
          docs_table = docs_table.iloc[0]
      
      docname = docs_table['Document']
      
      # If that first entry is unavailable,
      # log the failure and exit
      if str(docname) == 'nan':
          os.chdir('..')
          text = 'File with CIK: %s and Acc_No: %s is unavailable' % (cik, acc_no) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue       
      
      # If it is available, continue...
      
      # Request the file
      file = requests.get(doc_url_base % (cik, acc_no.replace('-', ''), docname))
      
      # If the request fails, log the failure and exit
      if file.status_code != 200:
          os.chdir('..')
          text = "Request failed with error code " + str(file.status_code) + \
                 "\nFailed URL: " + (doc_url_base % (cik, acc_no.replace('-', ''), docname)) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue
          
      # If it succeeds, keep going...
      
      # Save the file in appropriate format
      if '.txt' in docname:
          # Save text as TXT
          date = str(row['Filing Date'])
          filename = cik + '_' + date + '.txt'
          html_file = open(filename, 'a')
          html_file.write(file.text)
          html_file.close()
      else:
          # Save text as HTML
          date = str(row['Filing Date'])
          filename = cik + '_' + date + '.html'
          html_file = open(filename, 'a')
          html_file.write(file.text)
          html_file.close()
      
  # Move back to the main 10-Q directory
  os.chdir('..')
      
  return


def ScrapeS1(browse_url_base, filing_url_base, doc_url_base, cik, ticker, log_file_name):
  
  '''
  Scrapes all S-1's for a particular CIK from EDGAR.
  
  Parameters
  ----------
  browse_url_base : str
      Base URL for browsing EDGAR.
  filing_url_base : str
      Base URL for filings listings on EDGAR.
  doc_url_base : str
      Base URL for one filing's document tables
      page on EDGAR.
  cik : str
      Central Index Key.
  log_file_name : str
      Name of the log file (should be a .txt file).
      
  Returns
  -------
  None.
  
  '''
  
  # Check if we've already scraped this CIK
  try:
      directory = str(cik) + '_' + ticker
      os.mkdir(directory)
  except OSError:
      print("Already scraped CIK", cik)
      return
  
  # If we haven't, go into the directory for that CIK
  os.chdir(directory)
  
  print('Scraping CIK', cik)
  
  # Request list of S1 filings
  res = requests.get(browse_url_base % cik)
  
  # If the request failed, log the failure and exit
  if res.status_code != 200:
      os.chdir('..')
      os.rmdir(directory) # remove empty dir
      text = "Request failed with error code " + str(res.status_code) + \
             "\nFailed URL: " + (browse_url_base % cik) + '\n'
      WriteLogFile(log_file_name, text)
      return
  
  # If the request doesn't fail, continue...

  # Parse the response HTML using BeautifulSoup
  soup = bs.BeautifulSoup(res.text, "lxml")

  # Extract all tables from the response
  html_tables = soup.find_all('table')
  
  # Check that the table we're looking for exists
  # If it doesn't, exit
  if len(html_tables)<3:
      print("table too short")
      os.chdir('..')
      return
  
  # Parse the Filings table
  filings_table = pd.read_html(str(html_tables[2]), header=0)[0]
  filings_table['Filings'] = [str(x) for x in filings_table['Filings']]

  # Get only S-1 document filings
  filings_table = filings_table[filings_table['Filings'] == 'S-1']

  # If filings table doesn't have any S-1's, exit
  if len(filings_table)==0:
      os.chdir('..')
      return
  

  # Get accession number for each 10-K and 10-K405 filing
  filings_table['Acc_No'] = [x.replace('\xa0',' ')
                             .split('Acc-no: ')[1]
                             .split(' ')[0] for x in filings_table['Description']]

  # Iterate through each filing and 
  # scrape the corresponding document...
  for index, row in filings_table.iterrows():
      
      # Get the accession number for the filing
      acc_no = str(row['Acc_No'])
      
      # Navigate to the page for the filing
      docs_page = requests.get(filing_url_base % (cik, acc_no))
      
      # If request fails, log the failure
      # and skip to the next filing    
      if docs_page.status_code != 200:
          os.chdir('..')
          text = "Request failed with error code " + str(docs_page.status_code) + \
                 "\nFailed URL: " + (filing_url_base % (cik, acc_no)) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue
          
      # If request succeeds, keep going...
      
      # Parse the table of documents for the filing
      docs_page_soup = bs.BeautifulSoup(docs_page.text, 'lxml')
      docs_html_tables = docs_page_soup.find_all('table')
      if len(docs_html_tables)==0:
          continue
      docs_table = pd.read_html(str(docs_html_tables[0]), header=0)[0]
      docs_table['Type'] = [str(x) for x in docs_table['Type']]
      
      # Get the S-1 entries for the filing
      docs_table = docs_table[docs_table['Type'] == 'S-1']

      
      # If there aren't any 10-K or 10-K405 entries,
      # skip to the next filing
      if len(docs_table)==0:
          continue
      # If there are 10-K or 10-K405 entries,
      # grab the first document
      elif len(docs_table)>0:
          docs_table = docs_table.iloc[0]
      
      docname = docs_table['Document']
      
      # If that first entry is unavailable,
      # log the failure and exit
      if str(docname) == 'nan':
          os.chdir('..')
          text = 'File with CIK: %s and Acc_No: %s is unavailable' % (cik, acc_no) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue       
      
      # If it is available, continue...
      
      # Request the file
      file = requests.get(doc_url_base % (cik, acc_no.replace('-', ''), docname))
      
      # If the request fails, log the failure and exit
      if file.status_code != 200:
          os.chdir('..')
          text = "Request failed with error code " + str(file.status_code) + \
                 "\nFailed URL: " + (doc_url_base % (cik, acc_no.replace('-', ''), docname)) + '\n'
          WriteLogFile(log_file_name, text)
          os.chdir(directory)
          continue
      
      print("S-1 rquested and found for %s \n" % (ticker))  


      # If it succeeds, keep going...
      
      # Save the file in appropriate format
      if '.txt' in docname:
          # Save text as TXT
          date = str(row['Filing Date'])
          filename = cik + '_' + date + '_s1.txt'
          html_file = open(filename, 'a')
          html_file.write(file.text)
          html_file.close()
      else:
          # Save text as HTML
          date = str(row['Filing Date'])
          filename = cik + '_' + date + '_s1.html'
          html_file = open(filename, 'a')
          html_file.write(file.text)
          html_file.close()
      
  # Move back to the main S-1 directory
  os.chdir('..')
      
  return


def RunScrape10K(ticker_cik_df):

  '''
  Iterates over all companies to scrape 10ks.
  
  Parameters
  ----------
  ticker_cik_df : pandas df
    dataframe containing cik, ticker, and sic code for all public companies

      
  Returns
  -------
  None.

  '''
  
  # Define parameters
  browse_url_base_10k = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=10-K'
  filing_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s-index.html'
  doc_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s/%s'

  # Set correct directory
  os.chdir(PATHNAME_10K)

  # Initialize log file
  # (log file name = the time we initiate scraping session)
  time = strftime("%Y-%m-%d %Hh%Mm%Ss", gmtime())
  log_file_name = 'logs/log '+time+'.txt'
  with open(log_file_name, 'a') as log_file:
    log_file.close()

  # Iterate over CIKs and scrape 10-Ks
  for index, row in tqdm(ticker_cik_df[['cik','ticker']].iterrows()):
    Scrape10K(browse_url_base=browse_url_base_10k, 
      filing_url_base=filing_url_base_10k, 
      doc_url_base=doc_url_base_10k, 
      cik=row['cik'],
      ticker=row['ticker'],
      log_file_name=log_file_name)


def RunScrape10Q(ticker_cik_df):

  '''
  Wrapper function that iterates over all companies to scrape 10qs.
  
  Parameters
  ----------
  ticker_cik_df : pandas df
    dataframe containing cik, ticker, and sic code for all public companies
      
  Returns
  -------
  None.

  '''

  # Define parameters
  browse_url_base_10q = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=10-Q&count=1000'
  filing_url_base_10q = 'http://www.sec.gov/Archives/edgar/data/%s/%s-index.html'
  doc_url_base_10q = 'http://www.sec.gov/Archives/edgar/data/%s/%s/%s'

  # Set correct directory (fill this out yourself!)
  os.chdir(PATHNAME_10Q)

  # Initialize log file
  # (log file name = the time we initiate scraping session)
  time = strftime("%Y-%m-%d %Hh%Mm%Ss", gmtime())
  log_file_name = 'logs/log '+time+'.txt'
  log_file = open(log_file_name, 'a')
  log_file.close()

  # Iterate over CIKs and scrape 10-Qs
  for index, row in tqdm(ticker_cik_df[['cik','ticker']].iterrows()):
    Scrape10Q(browse_url_base=browse_url_base_10q, 
      filing_url_base=filing_url_base_10q, 
      doc_url_base=doc_url_base_10q, 
      cik=row['cik'],
      ticker=row['ticker'],
      log_file_name=log_file_name)



def RunScrapeS1(ticker_cik_df):

  '''
  Wrapper function that iterates over all companies to scrape 10qs.
  
  Parameters
  ----------
  ticker_cik_df : pandas df
    dataframe containing cik, ticker, and sic code for all public companies
      
  Returns
  -------
  None.

  '''

  # Define parameters 
  browse_url_base_s1 = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=S-1'
  filing_url_base_s1 = 'http://www.sec.gov/Archives/edgar/data/%s/%s-index.html'
  doc_url_base_s1 = 'http://www.sec.gov/Archives/edgar/data/%s/%s/%s'

  # Set correct directory (fill this out yourself!)
  os.chdir(PATHNAME_S1)

  # Initialize log file
  # (log file name = the time we initiate scraping session)
  time = strftime("%Y-%m-%d %Hh%Mm%Ss", gmtime())
  log_file_name = 'logs/log '+time+'.txt'
  log_file = open(log_file_name, 'a')
  log_file.close()

  # Iterate over CIKs and scrape 10-Ks
  for index, row in tqdm(ticker_cik_df[['cik','ticker']].iterrows()):
    ScrapeS1(browse_url_base=browse_url_base_s1, 
      filing_url_base=filing_url_base_s1, 
      doc_url_base=doc_url_base_s1, 
      cik=row['cik'],
      ticker=row['ticker'],
      log_file_name=log_file_name)


def RemoveNumericalTables(soup):
    
  '''
  Removes tables with >15% numerical characters.
  
  Parameters
  ----------
  soup : BeautifulSoup object
    Parsed result from BeautifulSoup.
      
  Returns
  -------
  soup : BeautifulSoup object
    Parsed result from BeautifulSoup
    with numerical tables removed.
      
  '''
  
  # Determines percentage of numerical characters
  # in a table
  def GetDigitPercentage(tablestring):
    if len(tablestring)>0.0:
      numbers = sum([char.isdigit() for char in tablestring])
      length = len(tablestring)
      return numbers/length
    else:
      return 1
  
  # Evaluates numerical character % for each table
  # and removes the table if the percentage is > 15%
  [x.extract() for x in soup.find_all('table') if GetDigitPercentage(x.get_text())>0.15]
  
  return soup

def RemoveTags(soup):
    
  '''
  Drops HTML tags, newlines and unicode text from
  filing text.
  
  Parameters
  ----------
  soup : BeautifulSoup object
    Parsed result from BeautifulSoup.
      
  Returns
  -------
  text : str
    Filing text.
      
  '''
  
  # Remove HTML tags with get_text
  text = soup.get_text()
  
  # Remove newline characters
  text = text.replace('\n', ' ')
  
  # Replace unicode characters with their
  # "normal" representations
  text = unicodedata.normalize('NFKD', text)
  
  return text

def ConvertHTML(cik):
  
  '''
  Removes numerical tables, HTML tags,
  newlines, unicode text, and XBRL tables.
  
  Parameters
  ----------
  cik : str
      Central Index Key used to scrape files.
  
  Returns
  -------
  None.
  
  '''
  
  # Look for files scraped for that CIK
  try: 
    os.chdir(cik)
  # ...if we didn't scrape any files for that CIK, exit
  except FileNotFoundError:
    print("Could not find directory for CIK", cik)
    return
      
  print("Parsing CIK %s..." % cik)
  parsed = False # flag to tell if we've parsed anything
  
  # Try to make a new directory within the CIK directory
  # to store the text representations of the filings
  try:
    os.mkdir('rawtext')
  # If it already exists, continue
  # We can't exit at this point because we might be
  # partially through parsing text files, so we need to continue
  except OSError:
    pass
  
  # Get list of scraped files
  # excluding hidden files and directories
  file_list = [fname for fname in os.listdir() if not (fname.startswith('.') | os.path.isdir(fname))]
  
  # Iterate over scraped files and clean
  for filename in file_list:
        
    # Check if file has already been cleaned
    new_filename = filename.replace('.html', '.txt')
    text_file_list = os.listdir('rawtext')
    if new_filename in text_file_list:
      continue
    
    # If it hasn't been cleaned already, keep going...
    
    # Clean file
    with open(filename, 'r') as file:
      parsed = True
      soup = bs.BeautifulSoup(file.read(), "lxml")
      soup = RemoveNumericalTables(soup)
      text = RemoveTags(soup)
      with open('rawtext/'+new_filename, 'w') as newfile:
        newfile.write(text)
  
  # If all files in the CIK directory have been parsed
  # then log that
  if parsed==False:
    print("Already parsed CIK", cik)
  
  os.chdir('..')
  return

def ConvertToText(ticker_cik_df):
  
  '''
  Removes numerical tables, HTML tags,
  newlines, unicode text, and XBRL tables.
  
  Parameters
  ----------
  ticker_cik_df : pandas df
    dataframe containing cik, ticker, and sic code for all public companies
  
  Returns
  -------
  None.
  
  '''

  # # For 10-Ks...
  # os.chdir(PATHNAME_10K)
  # # Iterate over CIKs and clean HTML filings
  # for cik in tqdm(ticker_cik_df['cik']):
  #   ConvertHTML(cik)

  # # For 10-Qs...
  # os.chdir(PATHNAME_10Q)
  # # Iterate over CIKs and clean HTML filings
  # for cik in tqdm(ticker_cik_df['cik']):
  #   ConvertHTML(cik)

  # For S-1s...
  os.chdir(PATHNAME_S1)
  # Iterate over CIKs and clean HTML filings
  for index, row in tqdm(ticker_cik_df[['cik','ticker']].iterrows()):
    cik = str(row['cik'])
    ticker = row['ticker']
    directory = cik + '_' + ticker
    ConvertHTML(directory)


def AggregateRawText(pathname, ticker_cik_df):

  '''
  Finds all of the raw text files created by ConvertToText
  and aggregates them into the single folder 'all_raw_text'
  located under the pathname.
  
  Parameters
  ----------
  pathname : str
    string containing the path to aggregate the text files from; should be s1, 10k, or 10q

  ticker_cik_df : pandas df
    dataframe containing cik, ticker, and sic code for all public companies
  
  Returns
  -------
  None.
  
  '''

  # Set the destination folder
  dest = os.path.join(pathname, 'all_raw_text')
  
  # print(dest)

  # Iterate over the folders and gather the text files
  for index, row in tqdm(ticker_cik_df[['cik','ticker']].iterrows()):
    cik = str(row['cik'])
    ticker = row['ticker']
    directory = cik + '_' + ticker
    
    # Go into the raw text folder for the company and recursively copy all files
    company_path = os.path.join(pathname, directory, 'rawtext')
   
    # print(company_path)
    # code.interact(local = locals())

    os.chdir(company_path)
    file_list = [fname for fname in os.listdir() if not (fname.startswith('.') | os.path.isdir(fname))]
    for file_name in file_list:
      full_file_name = os.path.join(company_path, file_name)
      if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)





# Run the code
ticker_cik_df = GetTickerCikDf()


# RunScrape10K(ticker_cik_df)
# code.interact(local = locals())
# RunScrape10Q(ticker_cik_df)
# code.interact(local=locals())
# RunScrapeS1(ticker_cik_df)
# code.interact(local = locals())
# ConvertToText(ticker_cik_df)

AggregateRawText(PATHNAME_S1, ticker_cik_df)

