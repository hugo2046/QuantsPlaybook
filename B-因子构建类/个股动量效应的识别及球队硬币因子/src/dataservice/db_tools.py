'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-06-20 13:44:39
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-06-20 13:58:20
Description: 
'''

from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker


from .config import config


# 构建一个链接数据库的类
class DBConn(object):
    def __init__(self, conn_type: str = "windows_conn_str",db_name:str="datacenter"):
        
        self.con = f"{getattr(config, conn_type)}/{db_name}"
        self.engine = create_engine(
            self.con,
            pool_size=25,
            pool_recycle=3600,
            pool_pre_ping=True,
        )
        self.auto_db_base = automap_base()  # 数据表必须有主键否则无法识别
        self.auto_db_base.prepare(self.engine, reflect=True)

    def connect(self):
        self.Session = sessionmaker(bind=self.engine)