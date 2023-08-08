"""
Author: Hugo
Date: 2022-03-07 21:45:58
LastEditTime: 2023-08-02 11:09:36
LastEditors: hugo2046 shen.lan123@gmail.com
Description: tuhsare自动延迟下载,防止频繁调取数据是报错
"""
import logging
import logging.handlers
# tuhsare自动延迟下载，防止频繁调取数据是报错
import time

import tushare as ts

from .config import TS_TOKEN


class TuShare:
    """tushare服务接口自动重试封装类,能够在接口超时情况下自动等待1秒然后再次发起请求,
    无限次重试下去，直到返回结果或者达到最大重试次数。
    """

    def __init__(self, token: str = TS_TOKEN, logger: bool = None, max_retry: int = 0):
        """构造函数,token:tushare的token;logger:日志对象,可以不传；
        max_retry:最大重试次数,默认为0意为无限重试,建议用10以上100以内。"""
        self.token = token
        if not logger:
            logger = logging.getLogger("TuShare")
            # CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s %(name)s %(pathname)s:%(lineno)d %(funcName)s %(levelname)s %(message)s"
            )
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
        self.logger = logger
        self.max_retry = max_retry
        ts.set_token(token)
        self.pro = ts.pro_api()

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            i = 0
            while True:
                try:
                    if name == "pro_bar":
                        m = getattr(ts, name, None)
                    else:
                        m = getattr(self.pro, name, None)
                    if m is None:
                        self.logger.error("Attribute %s does not exist.", name)
                        return None
                    else:
                        return m(*args, **kwargs)
                except Exception:
                    if self.max_retry > 0 and i >= self.max_retry:
                        raise
                    self.logger.exception(
                        "TuShare exec %s failed, args:%s, kwargs:%s, try again.",
                        name,
                        args,
                        kwargs,
                    )
                    time.sleep(1)
                i += 1

        return wrapper

