import numpy as np
import pandas as pd
import seaborn as sns
import calendar
import matplotlib.pyplot as plt
import time
import joblib

pd.set_option('display.max_columns', None)

start_time = time.time()
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 读取数据

data = pd.read_csv('data1.csv')
store = list(pd.read_csv('test.csv')['Store'].unique())

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 划分数据
print(list(data.columns))

data = data.loc[:,
       ['Id', 'Open', 'Promo', 'Sales', 'SchoolHoliday', 'Store', 'Year', 'Month', 'Day', 'DayOfYear', 'WeekOfYear', 'DateNum', 'DayOfWeek_1',
        'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'StateHoliday_a', 'Season_2', 'Season_3', 'close_open',
        '2014_Sales_avg_yeartoyear', 'StateHoliday_a_before']
       ]


def split(data):
    train = data.loc[(data['DateNum'] < 735678) & (data['Sales'] > 0)]
    val = data.loc[((data['DateNum'] >= 735678) & (data['DateNum'] < 735720)) & (data['Sales'] > 0)]
    test = data.loc[data['DateNum'] >= 735720]

    X_train = train.drop(['Sales', 'Id'], axis=1)
    X_val = val.drop(['Sales', 'Id'], axis=1)
    X_test = test.drop(['Sales', 'Id'], axis=1)

    y_train = np.log1p(train['Sales'].values)
    y_val = np.log1p(val['Sales'].values)
    return val, test, X_train, X_val, X_test, y_train, y_val


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 损失函数
def error(y_hat, y_true):
    error = np.mean(np.absolute(y_hat / y_true - 1))
    return error


def error_xbg(y_hat, y_true):
    y_hat = np.expm1(y_hat)
    y_true = np.expm1(y_true.get_label())
    return 'rmspe', error(y_hat, y_true)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 建模
import xgboost as xgb


# 模型训练
def train(val, test, X_train, X_val, X_test, y_train, y_val):
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_val, y_val)
    evals = [(dtrain, 'train'), (dvalid, 'val')]
    # evals = [(dtrain, 'train')]  # train all

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=35, verbose_eval=True, feval=error_xbg)
    y_val_predict = np.expm1(model.predict(xgb.DMatrix(X_val)))
    y_test_predict = np.expm1(model.predict(xgb.DMatrix(X_test)))
    # plt.figure(figsize=(90,90))
    # xgb.plot_importance(model)
    # plt.show()
    return y_val_predict, y_test_predict


# 对不同店铺分别训练
val_output = pd.DataFrame()
test_output = pd.DataFrame()

n = 0
for seed in [10, 290, 33, 68, 2030]:
    n += 1
    params = {'objective': 'reg:linear',
              'booster': 'gbtree',
              'silent': 1,
              'eta': 0.02,
              'max_depth': 10,
              'subsample': 0.5,
              'colsample_bytree': 0.5,
              'seed': seed,
              'tree_method': 'exact'
              }

    for s in store:
        print('第{}次{}门店-------------------------------------------------------------------------------------------'.format(n, s))
        data_s = data[data['Store'] == s]
        val, test, X_train, X_val, X_test, y_train, y_val = split(data_s)
        y_val_predict, y_test_predict = train(val, test, X_train, X_val, X_test, y_train, y_val)

        # val_a = pd.concat([val.reset_index(), pd.Series(y_val_predict, name='predict')], axis=1)
        test_a = pd.concat([test['Id'].reset_index(), pd.Series(y_test_predict, name='predict')], axis=1)
        # val_output = pd.concat([val_a, val_output], axis=0)
        test_output = pd.concat([test_a, test_output], axis=0)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 导出数据

# val_output.to_csv('val_data1_' + str(n) + '.csv', index=False)
test_output = test_output.groupby('Id')['predict'].mean()
test_output = pd.DataFrame(pd.Series(test_output, name='Sales'), dtype=np.int)
test_output = test_output.sort_index(ascending=True)
test_output.to_csv('sample_submission.csv', index=True)

end_time = time.time()
print('耗时：{:.1f}分钟'.format((end_time - start_time) / 60))
