# 说明

## 数据获取

这里我们使用[tushare]([Tushare数据](https://tushare.pro/))获取的标的为:黄金ETF(518880.SH)、纳指ETF(513100.SH)、创业板ETF(159915.SZ)、沪深300ETF(510300.SH),时间范围2014-01-01至2023-08-02,价格数据后复权。

可以cd到项目目录下使用该命令获取默认数据(价格数据为后复权)

```bash
python get_data.py dump_all
```

如要获取其他数据可以使用

```bash
python get_data.py dump_all --codes "510300.SH" --start_dt "2023-01-01" --end_dt "2023-08-02"
```

使用时需要修改ts_data_service文件中的config，将tushare的token改为自己的

## 模型

具体参看mlt_tsmom.ipynb文件
