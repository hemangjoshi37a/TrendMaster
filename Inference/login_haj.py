from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import urllib.parse as urlparse
from selenium.webdriver.chrome.options import Options
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
import pandas as pd
import datetime
import joblib
import pdb
import logging
logging.basicConfig(level=logging.ERROR)

# pip3 install selenium
# pip3 install urllib3

class ZerodhaAccessToken:
    def __init__(self):
        self.apiKey = 'API_KEY'
        self.apiSecret = 'API_SECRET'
        self.accountUserName = 'ACCOUNT_ID'
        self.accountPassword = 'ACCOUNT_PASSWORD'
        self.securityPin = 'ACCOUNT_PIN'

    def getaccesstoken(self):
        try:
            login_url = "https://kite.trade/connect/login?v=3&api_key={apiKey}".format(apiKey=self.apiKey)

            chrome_driver_path = "/usr/bin/chromedriver"
            options = Options()
#             options.add_argument('--headless') #for headless
            driver = webdriver.Chrome(chrome_driver_path, options=options)
            driver.get(login_url)
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="text"]')))\
                .send_keys(self.accountUserName)
            wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="password"]')))\
                .send_keys(self.accountPassword)
            wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]')))\
                .submit()
            wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="password"]'))).click()
            time.sleep(20)
            driver.find_element_by_xpath('//input[@type="password"]').send_keys(self.securityPin)
            wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))).submit()
            wait.until(EC.url_contains('status=success'))
            tokenurl = driver.current_url
            parsed = urlparse.urlparse(tokenurl)
            driver.close()
            return urlparse.parse_qs(parsed.query)['request_token'][0]
        except Exception as ex:
            print(ex)
            
# _ztoken = ZerodhaAccessToken()
# actual_token = _ztoken.getaccesstoken()
# print('access token : '+str(actual_token))
# kite = KiteConnect(api_key=_ztoken.apiKey)
# data = kite.generate_session(actual_token,api_secret=_ztoken.apiSecret)
# kite.set_access_token(data["access_token"])
# print('request token : '+str(data["access_token"]))
# joblib.dump(kite,'kitefile.p')
# kws = KiteTicker(_ztoken.apiKey, data["access_token"])
            
def auto_login():
    global kite,kws,data,_ztoken,actual_token
    _ztoken = ZerodhaAccessToken()
    actual_token = _ztoken.getaccesstoken()
    print('access token : '+str(actual_token))
    kite = KiteConnect(api_key=_ztoken.apiKey)
    data = kite.generate_session(actual_token,api_secret=_ztoken.apiSecret)
    kite.set_access_token(data["access_token"])
    print('request token : '+str(data["access_token"]))
    joblib.dump(kite,'kitefile.p')
    kws = KiteTicker(_ztoken.apiKey, data["access_token"])
    
    
def retry_autologin():
    try :
        print('Trying to login...')
        auto_login()
    except AttributeError as ee:
        retry_autologin()
        
# retry_autologin()
