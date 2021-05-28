import datetime
import shutil
import os
# import automagica.activities as all_method
import sys
import time
import base64
sys.path.append('..')
sys.path.append('.')
sys.path.append('../')
sys.path.append('..')
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
options = webdriver.ChromeOptions()
chrome_options = options
options.add_experimental_option('excludeSwitches', ['enable-automation'])
dic_for_window ={}
import pickle

from env_variable import *
import binascii
import io

'''
demo:
-p http://www.biying.com
'''
try:

    import argparse


    # all_method.openBrowser("http://www.baidu.com")

    # all_method.move_mouse_to_coordinates(10,10)

        # 最新版本传参说明.    更加简化了json格式.  字符串转译的还需要写/"      如果没有参数那么就传空""即可.
        # 改成写文件吧.不然 转义字符来回弄非常复杂. 前端把命令写入tmp.json文件中.

        # 后期并发情况,可以 通过argparse,传送一个json文件名.然后文件名前端保存时候加上时间措即可.
        # 现在先考虑单进程情况. 只是tmp.json写死.
    parser = argparse.ArgumentParser()
    parser.add_argument( "-p")
    args = parser.parse_args()
    if not args.p:
        args.p="http://www.biying.com"
    # print(args.url)
    # print(11111111)

    # "dic_for_window[{}]={}\n".format("browser.title", "browser.current_window_handle"),
    # browser='''
    # aaaaaaaaaaaaaaaio.BytesIO

    def create_driver():
        driver = webdriver.Chrome(options=chrome_options)
        with open(session_file, 'wb') as f:
            params = {"session_id": driver.session_id, "server_url": driver.command_executor._url}
            pickle.dump(params, f)
        driver.get(args.p)

        dic_for_window[driver.title]= driver.current_window_handle
        with open(dic_for_window_pickle, 'wb') as f:
            pickle.dump(dic_for_window, f)
        return driver


    if not os.path.exists(session_file):
        driver = create_driver()
    else:
        with open(session_file, 'rb') as f:
            params = pickle.load(f)
            try:
                options = webdriver.ChromeOptions()
                options.add_argument("headless")
                driver = webdriver.Remote(command_executor=params["server_url"],options=options)
                driver.quit()  # 退出start_session新开的空白浏览器
                driver.session_id = params["session_id"]
                # driver.execute_script('window.open("");')
                # driver.get(args.url)
                driver.get(args.p)
                dic_for_window[driver.title] = driver.current_window_handle
                with open(dic_for_window_pickle, 'wb') as f:
                    pickle.dump(dic_for_window, f)

                driver.switch_to.window(driver.window_handles[-1])
            except:
                driver = create_driver()

    print("{'code':'0','msg':'成功'}")







except:
    print("{'code':'1','msg':'错误'}")

