# 说明

## 数据获取

src\dataservice是数据获取所需脚本,这里主要是从我自己的数据库获取数据,仅需在src\dataservice中建立一个config.py写入自己的数据库链接即可:

```python
__all__ = "config"


class Config(object):
    def __init__(self) -> None:
        self.windows_conn_str: str = (
            "mysql+mysqlconnector://用户:密码@ip:端口号"
        )
        self.linux_conn_str: str = (
            "mysql+mysqlconnector://用户:密码@ip:端口号"
        )


config: Config = Config()
```

数据获在命令行中运行:

```bash
$cd 个股动量效应的识别及“球队硬币”因子 && python src\savedata2csv --start_date 2013-01-01 --end_date 2023-05-31
```

---

### 数据库表名及结构

数据主要从我自己数据库中获取,数据表结构如下:

**表名:daily**

| 名称       | 类型  | 描述             |
| :--------- | :---- | :--------------- |
| ts_code    | str   | 股票代码         |
| trade_date | str   | 交易日期         |
| open       | float | 开盘价           |
| high       | float | 最高价           |
| low        | float | 最低价           |
| close      | float | 收盘价           |
| pre_close  | float | 昨收价(前复权)   |
| change     | float | 涨跌额           |
| pct_chg    | float | 涨跌幅 （未复权) |
| vol        | float | 成交量 （手）    |
| amount     | float | 成交额 （千元）  |

**表名:adj_factor**

| 名称       | 类型  | 描述     |
| :--------- | :---- | :------- |
| ts_code    | str   | 股票代码 |
| trade_date | str   | 交易日期 |
| adj_factor | float | 复权因子 |

**表名:daily_basic**

| 名称            | 类型  | 描述                                   |
| :-------------- | :---- | :------------------------------------- |
| ts_code         | str   | TS股票代码                             |
| trade_date      | str   | 交易日期                               |
| close           | float | 当日收盘价                             |
| turnover_rate   | float | 换手率（%）                            |
| turnover_rate_f | float | 换手率（自由流通股）                   |
| volume_ratio    | float | 量比                                   |
| pe              | float | 市盈率（总市值/净利润， 亏损的PE为空） |
| pe_ttm          | float | 市盈率（TTM，亏损的PE为空）            |
| pb              | float | 市净率（总市值/净资产）                |
| ps              | float | 市销率                                 |
| ps_ttm          | float | 市销率（TTM）                          |
| dv_ratio        | float | 股息率 （%）                           |
| dv_ttm          | float | 股息率（TTM）（%）                     |
| total_share     | float | 总股本 （万股）                        |
| float_share     | float | 流通股本 （万股）                      |
| free_share      | float | 自由流通股本 （万）                    |
| total_mv        | float | 总市值 （万元）                        |
| circ_mv         | float | 流通市值（万元）                       |

## 数据转换

将csv转换为qlib所需结构在命令行运行如下命令:

```bash
$cd 个股动量效应的识别及“球队硬币”因子 && python src\dump_bin.py dump_all --csv_path data\cn_data --qlib_dir  data\qlib_data --date_field_name trade_date --exclude_fields code
```