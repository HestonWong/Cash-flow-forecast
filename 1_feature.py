import numpy as np
import pandas as pd
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.max_columns', None)

start_time = time.time()
# ----------------------------------------------------------------------------------------------------------------------
# 读取数据
train = pd.read_csv('train.csv', parse_dates=[2], low_memory=False).drop('Customers', axis=1)  # parse_dates 以日期类型导入
test = pd.read_csv('test.csv', parse_dates=[3], low_memory=False)
store = pd.read_csv('store.csv')
ratio = pd.read_csv('ratio.csv')

test.fillna(1, inplace=True)
store['CompetitionDistance'].fillna(np.max(store['CompetitionDistance']), inplace=True)
store.fillna(0, inplace=True)

data = pd.concat([train, test], axis=0, ignore_index=True, sort=True)
data = pd.merge(data, store, on='Store')


# ----------------------------------------------------------------------------------------------------------------------
# 特征工程

# 季节转换
def get_season(x):  # 添加对应的季节特征
    if x <= 3:
        return 1
    elif 3 < x <= 6:
        return 2
    elif 6 < x <= 9:
        return 3
    else:
        return 4


def features_create(data):
    # 日期转换
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['WeekOfYear'] = data['Date'].dt.weekofyear
    data['DateNum'] = data['Year'] * 365 + data['Month'] * 30.5 + data['Day']
    data['Season'] = data['Month'].apply(lambda x: get_season(x))
    data.drop('Date', axis=1, inplace=True)

    # 新增'CompetitionOpen'和'PromoOpen'特征,计算某天某店铺的竞争对手已营业时间和店铺已促销时间，用月为单位表示
    data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (data['Month'] - data['CompetitionOpenSinceMonth'])
    data['PromoOpen'] = 12 * (data['Year'] - data['Promo2SinceYear']) + (data['WeekOfYear'] - data['Promo2SinceWeek']) / 4.286
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceYear', 'Promo2SinceWeek'], axis=1, inplace=True)

    # 热编码
    data[['DayOfWeek', 'Season']] = data[['DayOfWeek', 'Season']].astype(np.object)
    data1 = pd.get_dummies(data)
    data = pd.merge(data, data1)
    data[['DayOfWeek', 'Season']] = data[['DayOfWeek', 'Season']].astype(np.int)
    data = data.drop(['StateHoliday', 'StoreType', 'Assortment'], axis=1)

    # 判断昨天是否关门和关门累积次数（连续关门代表装修门店，则有可能开店后会有开门红）
    data_new = pd.DataFrame()
    data['close_open'] = 0
    store_num = data['Store'].unique()

    for i in store_num:
        data_s = data[data['Store'] == i].sort_values('DateNum').reset_index(drop=True)  # 注意重置索引，不然data_s.loc[index, 'close_open'] = close_count会报错
        n = data_s.shape[0]
        close_count = 0
        for index in range(0, n):
            if data_s.loc[index, 'Open'] == 0:  # 当open状态一直处于关店时候，那么就会累积计算关店的次数
                close_count += 1
            elif close_count == 1:  # 当open状态从关店到开店时候，那么开店的第一天就会记录之前累积关店的次数，计数会重置
                data_s.loc[index, 'close_open'] = 1
                close_count = 0
            elif close_count > 1:  # 当open状态从关店到开店且次数大于1时候，那么开店的第一天就会记录之前累积关店的次数，计数会重置
                data_s.loc[index:index + int(close_count / 3), 'close_open'] = close_count
                close_count = 0

        data_new = pd.concat([data_new, data_s], axis=0)

    # 判断是否处于StateHoliday_a前后（有些假期前后会出现销售增长的情况）
    data_new1 = pd.DataFrame()
    data_new['StateHoliday_a_before'] = 0
    for i in store_num:
        data_s = data_new[data_new['Store'] == i].sort_values('DateNum').reset_index(drop=True)
        n = data_s.shape[0]
        for index in range(0, n):
            if data_s.loc[index, 'StateHoliday_a'] == 1:  # 当假期为a时候，则新增一个字段用来标记此假期前后的日子
                data_s.loc[index - 2:index - 1, 'StateHoliday_a_before'] = 1  # 标记假期a前两天为1
                data_s.loc[index + 1, 'StateHoliday_a_before'] = 1  # 标记假期a后天为1

        data_new1 = pd.concat([data_new1, data_s], axis=0)

    return data_new1


data = features_create(data)
data = pd.merge(data, ratio, how='left', on=['Store', 'Month'])  # 添加去年同比情况（去年同比增长的话，今年大概率也会增长）
data = data.sort_values('DateNum').reset_index(drop=True)

# 多项式转换
# poly = PolynomialFeatures(3, include_bias=False)
# poly_list = ['Day', 'DayOfYear']
# poly_df = pd.DataFrame(poly.fit_transform(data[poly_list]), columns=poly.get_feature_names())
# data = pd.concat([poly_df, data], axis=1).drop(['x0', 'x1'], axis=1)

print(data.shape)
print(list(data.columns))
data.to_csv('data1.csv', index=False)
end_time = time.time()
print('耗时：{:.1f}分钟'.format((end_time - start_time) / 60))
