'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-06-21 09:24:16
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-06-22 10:37:44
Description: 加载设置
'''
import json
import os
import sys
from pathlib import Path

cur_root = Path(__file__).absolute().parent

os.chdir(cur_root)

sys.path.append(str(cur_root.parent))

# print(os.listdir('..'))
__all__ = ['ts_token']

with open(r'config.json', 'r') as file:

    config = json.loads(file.read())

ts_token = config['ts_token']
