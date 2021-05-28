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
from lxml import etree
'''
demo:不加参数.   直接彻底关闭chrome
'''
import requests
try:

    import argparse


    # all_method.openBrowser("http://www.baidu.com")

    # all_method.move_mouse_to_coordinates(10,10)

        # 最新版本传参说明.    更加简化了json格式.  字符串转译的还需要写/"      如果没有参数那么就传空""即可.
        # 改成写文件吧.不然 转义字符来回弄非常复杂. 前端把命令写入tmp.json文件中.

        # 后期并发情况,可以 通过argparse,传送一个json文件名.然后文件名前端保存时候加上时间措即可.
        # 现在先考虑单进程情况. 只是tmp.json写死.
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, nargs='+')
    args = parser.parse_args()
    aaa=requests.get(args.p[0])
    aaa.encoding = 'utf-8'
    print(aaa.text)
    aaa=aaa.text
    selector = etree.HTML(aaa)
    tmp=selector.xpath(args.p[1])
    print(        "{"+"'code':'0','msg':'{}'".format(tmp)+"}"    )







except:
    print("{'code':'1','msg':'错误'}")

