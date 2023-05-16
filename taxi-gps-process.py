# 用taxi gps数据计算每一辆出租车在一天内的收入
# 本文件为数据预处理,根据出租车原始GPS数据得到相应可处理数据

# 分析：输入为TAXIGPS数据。输出的形式,应该是：每条订单,收入。
# 为了后续的分析,在输出的基础上,需要再增加：
# 订单号,收入,车牌号,订单开始时间,订单结束时间,订单上车坐标,订单下车坐标,行驶里程

import pandas as pd

# 读取数据,为data-sample文件夹下的TaxiData-Sample
data2 = pd.read_csv(r'data-sample/TaxiData-Sample',header = None)
# 给数据命名列
data2.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat', 'OpenStatus', 'Speed']

data2=data2.sort_values(by=['VehicleNum','Stime'])

# 去除重复数据,数据清洗得到表1,字段与原始数据一致
data2 = data2[-((data2['OpenStatus'].shift(1)== data2['OpenStatus'].shift(-1))&\
      (data2['OpenStatus'].shift(1)!= data2['OpenStatus'])&\
      (data2['VehicleNum'].shift(1)==data2['VehicleNum'].shift(-1))&\
      (data2['VehicleNum']==data2['VehicleNum'].shift(1)))]

# 将时间字符串转换为pd的时间格式,后面可以轻松的做加减
data2['Stime'] = pd.to_datetime(data2['Stime'])

# 订单号生成
data2['OpenStatus1'] = data2['OpenStatus'].shift()
data2['VehicleNum1'] = data2['VehicleNum'].shift()

data2['Lng1'] = data2['Lng'].shift()
data2['Lat1'] = data2['Lat'].shift()
data2['Stime1'] = data2['Stime'].shift()

data2['StatusChange'] = data2['OpenStatus1']-data2['OpenStatus']

# 筛选出车辆状态变化的数据
data2 = data2[data2['OpenStatus1']==1]
data2['orderid'] = data2['StatusChange'].cumsum()
print(data2.head(30))

# 对数据进行处理,得到表2，字段为：订单号,轨迹点经纬度,轨迹点的时间（订单号需要自己定义）,速度（用来计算候时时长）
table2 = data2[['orderid','Lng','Lat','Stime','Speed']]

# 表2 计算里程与候车时间

# 定义计算路径长度函数

from math import pi
import numpy as np
def getdistance(lon1, lat1, lon2, lat2): # 经度1,纬度1,经度2,纬度2 （十进制度数）输入为DataFrame的列
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(lambda r:r*pi/180, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(a**0.5) 
    r = 6371 # 地球平均半径,单位为公里
    return c * r * 1000


# 生成订单的里程
table3 = table2.copy()
table3['Lng1'] = table3['Lng'].shift(-1)
table3['Lat1'] = table3['Lat'].shift(-1)
table3['orderid1'] = table3['orderid'].shift(-1)
table3['Stime1'] = table3['Stime'].shift(-1)

table3 = table3[(table3['orderid1'] == table3['orderid'])]
# 计算每个点与下一个点的距离
lon1 = table3['Lng']
lat1 = table3['Lat']
lon2 = table3['Lng1']
lat2 = table3['Lat1']
table3['distance'] = getdistance(lon1, lat1, lon2, lat2)

# 计算每个点与下一个点的时间差
table3['interval'] = (table3['Stime1']-table3['Stime']).apply(lambda r:r.seconds)

# 集计得到出租车路径长度
orderlenth = table3[['orderid','distance']].groupby('orderid').sum().reset_index()
# 集计得到出租车等待时间
waittime = table3[table3['distance']==0][['orderid','interval']].groupby('orderid').sum()

# 连接表2与表3成为表3，字段为：订单号,订单的里程,候时时长（已计算每个订单的里程和候时时长）
table3 = pd.merge(orderlenth,waittime,on = 'orderid',how='left')

# 表4的字段：订单号,车牌号,订单开始时间,订单结束时间,订单上车坐标,订单下车坐标，整理出每个订单的其他信息
# 将每一个出行的起点与终点提取出来
o = data2.iloc[:1].append(data2[data2['StatusChange']==1])
d = data2[(data2['StatusChange']==1).shift(-1).fillna(False)]
table4 = o.append(d).sort_values(by = ['orderid','Stime'])[['orderid','VehicleNum','Stime','Lng','Lat']]

# 加一列isd,如果该行为起点,则isd=0,如果该行为终点,则isd=1
table4['isd'] = [i%2 for i in range(len(table4))]

# 把O与D的信息放在同一行
table4['Etime'] = table4['Stime'].shift(-1)
table4['ELng'] = table4['Lng'].shift(-1)
table4['ELat'] = table4['Lat'].shift(-1)
table4 = table4[(table4['isd']==0)&(-table4['Etime'].isnull())]

# 连接表3与表4成为表5，连接两个表,再计算订单收入
table5 = pd.merge(table4,table3,on = 'orderid',how = 'left')

# 根据深圳出租车规则,每天的23时至次日凌晨6时为夜间
table5['isnight'] = (table5['Stime'].apply(lambda r:r.hour)<6)|(table5['Stime'].apply(lambda r:r.hour)>=23)

# 计价：
# 1 起步价：首2公里11.00元;  
# 2 里程价：超过2公里部分,每公里2.40元;  
# 3 返空费：每天的23时至次日凌晨6时,超过25公里部分,每公里按上述里程价的30%加收返空费：  
# 4 夜间附加费：夜间起步价16元,每天的23时至次日凌晨6时,按上述起步价和里程价的20%加收夜间附加费;  
# 5 候时费：每分钟0.80元;  
# 6 大件行李费：体积超过0.2立方米、重量超过20公斤的大件行李，每件0.50元。无法分析这项数据，略去。
table5['起步价'] = table5['isnight']*(16-11)+11
table5['里程价'] = ((table5['distance']-2000)>0)*(table5['distance']-2000)*2.4/1000
table5['返空费'] = table5['isnight']*((table5['distance']-25000)>0)*((table5['distance']-25000)*2.4*0.3/1000)
table5['夜间附加费'] = table5['isnight']*((table5['distance']-2000)>0)*(table5['distance']-2000)*2.4*0.2/1000
table5['候时费'] = table5['interval']/60*0.8
table5['price'] = table5['起步价'] + table5['里程价'] + table5['返空费'] + table5['夜间附加费'] + table5['候时费']

# 筛选掉没有计算出来价格的数据
table5 = table5[-table5['price'].isnull()]

# 保存成csv文件
table5.to_csv(r'data-sample/taxi-price-new.csv',index = None)
print(table5.head(30))