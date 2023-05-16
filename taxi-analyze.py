# 用taxi gps出租车原始GPS数据计算每一辆出租车在一天内的收入，分析出租车手的策略
# 本文件为数据分析，根据taxi-gps-process数据预处理得到的数据进行分析
# 解决问题：
# 1、出租车一天的收入
# 2、去哪接客挣钱多？
# 3、什么时候接客订单多？什么时候接客单笔订单价格高？
# 4、收入最高的车手与收入中等的车手，挣钱方式有什么不一样吗
# 5、高收入者比中收入者勤奋吗
# 6、高收入者与中收入者接客地点差在哪？
# 7、高收入与中收入者OD的差别
# 8、高收入者与中收入者接客时间差在哪？

import pandas as pd

# 读取price数据
table5 = pd.read_csv(r'data-sample/taxi-price.csv')
# 通过每个订单的价格，进行分析并绘制，剔除异常数据
# 计算订单用时
table5['interval'] = (pd.to_datetime(table5['Etime']) -pd.to_datetime(table5['Stime'])).apply(lambda r:r.seconds)
# 运营车速
table5['speed'] = (table5['distance']/table5['interval'])*3.6
# 将运营车速太快的筛掉
table5 = table5[table5['speed']<80].copy()
# 每分钟平均收入多少
table5['price_per_minutes'] = table5['price']/(table5['interval']/60)
# 输出table5的长度，即有多少条数据
print(len(table5))

# 集计收入
df0 = table5.groupby('VehicleNum')['price'].sum()

# 定义计算路径长度函数
from math import pi
import numpy as np

# 这里我们考虑的只是订单的收入，分析收入还需要考虑油费，跑的距离越长，油费就越高
def getdistance(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）输入为DataFrame的列
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
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000

# 计算出租车行驶的距离
# 读取原始数据
data = pd.read_csv(r'data-sample/TaxiData-Sample',header = None)
# 给数据命名列
data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat', 'OpenStatus', 'Speed']
data = data.sort_values(by = ['VehicleNum', 'Stime'])

# 将时间字符串转换为pd的时间格式，后面可以轻松的做加减
data['Stime'] = pd.to_datetime(data['Stime'])

# 清洗OpenStatus异常的数据，并输出清洗前后的数据长度
print('清洗OpenStatus异常的数据前',len(data))
data = data[-((data['OpenStatus'].shift(-1) == data['OpenStatus'].shift())&
(data['OpenStatus'].shift(-1) != data['OpenStatus'])&
(data['VehicleNum'].shift(-1) == data['VehicleNum'].shift())&
(data['VehicleNum'].shift(-1) == data['VehicleNum']))]
print('清洗OpenStatus异常的数据后',len(data))

# 定义计算路径长度
data['Lng1'] = data['Lng'].shift(-1)
data['Lat1'] = data['Lat'].shift(-1)
data['Stime1'] = data['Stime'].shift(-1)
data['VehicleNum1'] = data['VehicleNum'].shift(-1)

# 计算每个点与下一个点的距离
lon1 = data['Lng']
lat1 = data['Lat']
lon2 = data['Lng1']
lat2 = data['Lat1']
data['distance'] = getdistance(lon1, lat1, lon2, lat2)

# 计算每个点与下一个点的时间差
data['interval'] = (data['Stime1']-data['Stime']).apply(lambda r:r.seconds)

# 车辆速度
data['speed'] = (data['distance']/data['interval'])*3.6
data = data[data['VehicleNum1'] == data['VehicleNum']]
# 备份data份数据
datapro = data
datapro2 = data

# 计算每辆车的驾驶距离
disagg = data.groupby(['VehicleNum'])['distance'].sum().reset_index()

# 小汽车每百公里大概在35-75元左右不等，我们统一定为每公里0.75元油费
disagg['cost'] = (disagg['distance']/1000)*0.75
df0 = pd.merge(df0.reset_index(),disagg,on = 'VehicleNum')
df0['income'] = df0['price']-df0['cost']

# TODO 问题0：出租车一天的收入

# 用pandas自带hist绘制收入直方图
import matplotlib.pyplot as plt
fig = plt.figure(1,(12,3),dpi = 100)    
ax1 = plt.subplot(121)
plt.sca(ax1)
df0['income'].hist(ax = ax1,bins = 20)
plt.ylabel('Count')
plt.xlabel('income')
plt.xlim(0,df0['income'].quantile(1))
plt.title('Histogram of income')

import seaborn as sns
# 用seaborn绘制kdeplot核密度分布
ax2 = plt.subplot(122)
plt.sca(ax2)
sns.kdeplot(df0['income'],ax = ax2,label = 'Income')
plt.ylabel('Kernel density')
plt.xlabel('Income')
plt.xlim(0,df0['income'].quantile(1))
plt.title('Kdeplot of income')
plt.show()
fig.savefig('./images/income.png')

# TODO 问题一：去哪接客挣钱多？

# 用pandas自带hist绘制直方图
fig = plt.figure(1,(12,3),dpi = 100)    
ax1 = plt.subplot(121)
plt.sca(ax1)
table5['price_per_minutes'].hist(ax = ax1,bins = 50)
plt.ylabel('Count')
plt.xlabel('Price per minutes')
plt.title('Histogram of price per minutes')
plt.xlim(0,table5['price_per_minutes'].max())

# 用seaborn绘制kdeplot核密度分布
ax2 = plt.subplot(122)
plt.sca(ax2)
sns.kdeplot(table5['price_per_minutes'],ax = ax2,label = 'Income')
plt.ylabel('Count')
plt.xlabel('Price per minutes')

plt.xlim(0,table5['price_per_minutes'].quantile(1))
plt.title('Kdeplot of price per minutes')
plt.show()
fig.savefig('./images/price-per-minutes.png')

# 绘制一下热力图，看看在出租车订单起点处平均价格的空间分布
# 栅格数据
# 读取shapefile文件
import geopandas
shp = r'shapefile/grid/grid.shp'
grid = geopandas.GeoDataFrame.from_file(shp,encoding = 'gbk')

# 计算每个格子的平均收入
data = table5[['Lng','Lat','price_per_minutes']].copy()

# 经纬度小数点保留三位小数
import math
# 划定栅格划分范围
lon1 = 113.75194
lon2 = 114.624187
lat1 = 22.447837
lat2 = 22.864748

latStart = min(lat1, lat2);
lonStart = min(lon1, lon2);

# 定义栅格大小(单位m)
accuracy = 500;

# 计算栅格的经纬度增加量大小▲Lon和▲Lat
deltaLon = accuracy * 360 / (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360));
deltaLat = accuracy * 360 / (2 * math.pi * 6371004);
data['LONCOL'] = ((data['Lng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
data['LATCOL'] = ((data['Lat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')

data = data.groupby(['LONCOL','LATCOL'])['price_per_minutes'].mean().reset_index()

# 将集计的结果与栅格的geopandas执行merge操作
gridtoplot = pd.merge(grid,data,on = ['LONCOL','LATCOL'])
gridtoplot.head(5)

import plot_map
# 绘制地理分布图
fig = plt.figure(1,(10,10),dpi = 100)  
ax = plt.subplot(111)
plt.sca(ax)
fig.tight_layout(rect = (0.05,0.1,1,0.9))
bounds = [113.7, 22.42, 114.3, 22.8]
#绘制地图
plot_map.plot_map(plt,bounds,zoom = 12,style = 4)

# 设置colormap的数据
import matplotlib
vmax = gridtoplot['price_per_minutes'].quantile(0.99)
# 设定一个标准化的工具，设定OD的colormap最大最小值，他的作用是norm(count)就会将count标准化到0-1的范围内
norm = matplotlib.colors.Normalize(vmin=0,vmax=vmax)

# 设定colormap的颜色
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#9DCC42','#FFFE03','#F7941D','#E9420E','#FF0000'], 256)

# 将gridtoplot这个geodataframe进行绘制
# 用gridtoplot.plot，设定里面的参数是column = 'count'，以count这一列来绘制。参数cmap = cmap设定它的颜色
gridtoplot.plot(ax = ax,column = 'price_per_minutes',cmap = cmap,vmax = vmax,vmin = 0)

# 不显示坐标轴
plt.axis('off')    
# 然后要把镜头调整回到深圳地图那，不然镜头就在imshow处
ax.set_xlim(113.6,114.8)
ax.set_ylim(22.4,22.9)
# 绘制colorbar
plt.imshow([[0,vmax]], cmap=cmap)
# 设定colorbar的大小和位置
cax = plt.axes([0.13, 0.32, 0.02, 0.3])
plt.colorbar(cax=cax)
plt.show()
fig.savefig('./images/heatmap-taxi-price.png')


# TODO 问题二：什么时候接客订单多？什么时候接客单笔订单价格高？

table5['Shour'] = pd.to_datetime(table5['Stime']).apply(lambda r:r.hour)
# 每小时订单数
df1 = table5.groupby(['Shour'])['VehicleNum'].count().reset_index()
# 每小时平均订单价格
df2 = table5.groupby(['Shour'])['price_per_minutes'].mean().reset_index()
fig = plt.figure(1,(12,3),dpi = 100)    

#创建一个子图
ax1 = plt.subplot(121)
plt.sca(ax1)

#绘制折线图
plt.plot(df1['Shour'],df1['VehicleNum'],label = 'count')
plt.legend()
plt.title('Num. of orders per hour')
plt.xticks(range(24),df1['Shour'])
plt.ylim(0,700)
#创建另一个子图
ax2 = plt.subplot(122)
plt.sca(ax2)

#绘制折线图
plt.plot(df1['Shour'],df2['price_per_minutes'],label = 'price')
plt.legend()
plt.title('Average price per order')
plt.xticks(range(24),df1['Shour'])
plt.ylim(0,3)
plt.show()
fig.savefig('./images/nums-of-orders-per-hour.png')


# TODO 问题三：收入最高的车手与收入中等的车手，挣钱方式有什么不一样吗
highincome = df0[df0['income']>df0['income'].quantile(0.8)]['VehicleNum']
midincome = df0[(df0['income']>df0['income'].quantile(0.4))&(df0['income']<df0['income'].quantile(0.6))]['VehicleNum']

# 提取两组车手的订单
highincome_order = pd.merge(table5,highincome,on = 'VehicleNum')
midincome_order = pd.merge(table5,midincome,on = 'VehicleNum')

# 观察高收入中收入群体每天的订单数量，与订单每分钟收入分布情况
# 高收入中收入群体每天的订单数量
fig = plt.figure(1,(7,3),dpi = 100)    
ax = plt.subplot(121)
plt.sca(ax)
plt.boxplot([highincome_order.groupby('VehicleNum')['orderid'].count(),
midincome_order.groupby('VehicleNum')['orderid'].count()])
plt.xticks([1,2],['High income','Middle income'])
plt.ylabel('Num. of orders')

#订单每分钟收入分布情况
ax = plt.subplot(122)
plt.sca(ax)
plt.boxplot([highincome_order['price_per_minutes'],
midincome_order['price_per_minutes']])
plt.xticks([1,2],['High income','Middle income'])
plt.ylim(-0.1,4)
plt.ylabel('Price per minutes')
plt.show()
fig.savefig('./images/Price-per-minutes.png')

'''
问题1-3结论:
图一可以看到，高收入车手每日接单数量说比中收入车手多的，也就是，高收入主要靠勤奋
图二可看到，高收入车手的订单的每分钟平均收入也要稍微比中收入车手高
也就是说，高收入车手靠的不仅仅是勤奋，还有技巧

那么，高收入车手接的单为什么会稍微贵一点呢？猜想可能的原因有两种
猜想：
1.高收入车手每天非常勤奋，根据深圳出租价格特点，可能半夜也出来接单，这样就比较贵  
2.高收入车手存在拒载现象，或者只在特定区域接单，避免接到便宜的单，单价较贵  

下面为差距分析：
'''

# TODO 问题四：高收入者比中收入者勤奋吗
# 计算高收入群体和中收入群体开车总路程，空驶总路程，载客总路程

# 计算每辆车的空驶距离和载客距离，报错KeyError: 'VehicleNum'
disagg = datapro2.groupby(['VehicleNum','OpenStatus'])['distance'].sum().reset_index()

# 整理数据到一个DataFrame中
highagg = pd.merge(highincome,disagg, on = 'VehicleNum')
highagg['Index'] = 'High income'
highagg['distance'] = highagg['distance']/1000
midagg = pd.merge(midincome,disagg, on = 'VehicleNum')
midagg['Index'] = 'Middle income'
midagg['distance'] = midagg['distance']/1000

# 绘图
fig = plt.figure(1,(10,5),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
sns.boxplot(x="OpenStatus", y="distance",hue="Index", data=pd.concat([highagg,midagg]),ax = ax)
plt.ylabel('Travel distance $(km)$')
plt.ylim(0,700)
plt.xticks([0,1],['Idle','Delivery'])
plt.show()
fig.savefig('./images/travel-distance.png')

'''
很有意思的结论出来了：

1.高收入者一天中空载的行驶距离与中等收入者处于同一个水平，但载客距离则明显高一个水平，
也就是说，高收入者虽然开车的总距离长，但是多的距离都在载客距离上！    
2.高收入者的空驶距离和载客距离的方差都比中等收入者小，
说明他们更能够控制自己每天的行驶距离，也就是，开车更稳更有技巧
'''

# TODO 问题五：高收入者比中收入者开车更有技巧吗

# 我们观察一下每小时高收入群体和中等收入群体的订单收入情况  
# 将高收入和中等收入的订单打上标签，合并一起，后面可以用seaborn绘制分组的boxplot

high_hour = highincome_order[['Stime','price_per_minutes']].copy()
high_hour['Shour'] = pd.to_datetime(highincome_order['Stime']).apply(lambda r:r.hour)
high_hour['Index'] = 'High income'
mid_hour =  midincome_order[['Stime','price_per_minutes']].copy()
mid_hour['Shour'] = pd.to_datetime(midincome_order['Stime']).apply(lambda r:r.hour)
mid_hour['Index'] = 'Middle income'
hourdata = pd.concat([high_hour,mid_hour])

# 绘图
fig = plt.figure(1,(10,5),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
sns.boxplot(x="Shour", y="price_per_minutes",hue="Index", data=hourdata,ax = ax)
plt.ylabel('Price per minutes')
plt.xlabel('Order time')
plt.ylim(0,7)
plt.show()

# 统计高收入者和低收入者每小时平均的订单数量
high_hour =  (highincome_order.groupby(pd.to_datetime(highincome_order['Stime']
                  ).apply(lambda r:r.hour))['price'].count()
/len(highincome)).reset_index().rename(columns = {'price':'avg_count'})
high_hour['Index'] = 'High income'


mid_hour =  (midincome_order.groupby(pd.to_datetime(midincome_order['Stime']
                  ).apply(lambda r:r.hour))['price'].count()
/len(midincome)).reset_index().rename(columns = {'price':'avg_count'})
mid_hour['Index'] = 'Middle income'

# 绘图
fig = plt.figure(1,(10,5),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
sns.lineplot(x="Stime", y="avg_count",hue="Index", data=pd.concat([high_hour,mid_hour]),ax = ax)
plt.ylabel('Average num. of orders per driver')
plt.xlabel('Order time')
plt.ylim(0,4)
plt.xticks(range(24),range(24))
plt.show()
fig.savefig('./images/avg-count.png')

#统计高收入者和低收入者每小时平均的订单数量
high_hour =  (highincome_order.groupby(pd.to_datetime(highincome_order['Stime']
                  ).apply(lambda r:r.hour))['price'].count()
/len(highincome)).reset_index().rename(columns = {'price':'avg_count'})
high_hour['Index'] = 'High income'
high_hour['avg_count'] = high_hour['avg_count']/high_hour['avg_count'].sum()
mid_hour =  (midincome_order.groupby(pd.to_datetime(midincome_order['Stime']
                  ).apply(lambda r:r.hour))['price'].count()
/len(midincome)).reset_index().rename(columns = {'price':'avg_count'})
mid_hour['Index'] = 'Middle income'
mid_hour['avg_count'] = mid_hour['avg_count']/mid_hour['avg_count'].sum()
#绘图
fig = plt.figure(1,(10,5),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
sns.lineplot(x="Stime", y="avg_count",hue="Index", data=pd.concat([high_hour,mid_hour]),ax = ax)
plt.ylabel('Average num. of orders per driver(percentage %)')
plt.xlabel('Order time')
plt.xticks(range(24),range(24))
plt.show()
fig.savefig('./images/avg-count-percentage.png')
# 得到结论：高收入者比中收入者更倾向于在夜间活动

# 统计高收入者和低收入者每小时订单的出行距离
high_hour =  highincome_order[['Stime','distance']].copy()
high_hour['Shour'] = pd.to_datetime(highincome_order['Stime']).apply(lambda r:r.hour)
high_hour['Index'] = 'High income'
mid_hour =  midincome_order[['Stime','distance']].copy()
mid_hour['Shour'] = pd.to_datetime(midincome_order['Stime']).apply(lambda r:r.hour)
mid_hour['Index'] = 'Middle income'
hourdata = pd.concat([high_hour,mid_hour])
hourdata['distance'] = hourdata['distance']/1000

#绘图
fig = plt.figure(1,(10,5),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
sns.boxplot(x="Shour", y="distance",hue="Index", data=hourdata,ax = ax)
plt.ylabel('Distance $(km)$')
plt.ylim(0,80)
plt.xlabel('Order time')
plt.show()
fig.savefig('./images/distance-status.png')

# TODO 问题六：高收入者与中收入者接客地点差在哪？
data_list = []
# 高收入 计算每个格子的平均收入
data = highincome_order[['Lng','Lat','orderid']].copy()

# 删格化+集计
data['LONCOL'] = ((data['Lng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
data['LATCOL'] = ((data['Lat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')
data = data.groupby(['LONCOL','LATCOL'])['orderid'].count().reset_index()
data['orderid'] = data['orderid']/data['orderid'].sum()
data = data.rename(columns = {'orderid':'highincome_rate'})
data_list.append(data)

# 中收入：计算每个格子的平均收入
data = midincome_order[['Lng','Lat','orderid']].copy()

# 删格化+集计
data['LONCOL'] = ((data['Lng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
data['LATCOL'] = ((data['Lat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')
data = data.groupby(['LONCOL','LATCOL'])['orderid'].count().reset_index()
data['orderid'] = data['orderid']/data['orderid'].sum()
data = data.rename(columns = {'orderid':'midincome_rate'})

data_list.append(data)

# 将高收入和中收入的分布比例merge
data = pd.merge(data_list[0],data_list[1],on = ['LONCOL','LATCOL'],how = 'outer').fillna(0)

# 计算每个格子接客数量比例差距
data['gap'] = data['highincome_rate']-data['midincome_rate']

# 栅格数据 读取shapefile文件
shp = r'shapefile/grid/grid.shp'
grid = geopandas.GeoDataFrame.from_file(shp,encoding = 'gbk')

# 将集计的结果与栅格的geopandas执行merge操作
gridtoplot = pd.merge(grid,data,on = ['LONCOL','LATCOL'])
gridtoplot.head(5)

# 绘制GIS图
fig = plt.figure(1,(10,8),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
bounds = [113.6,22.4,114.8,22.9]

# plot_map绘制地图底图
# 绘制行政区划
shp = r'shapefile/sz.shp'
xzqh = geopandas.GeoDataFrame.from_file(shp,encoding = 'utf-8')
xzqh.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0.2),linewidths=0.5)

# 设置colormap的数据
vmax = max(gridtoplot['gap'].max(),-gridtoplot['gap'].min())
# 设定colormap的颜色
cmapname = 'seismic'
cmap = matplotlib.cm.get_cmap(cmapname)

# 将gridtoplot这个geodataframe进行绘制
gridtoplot.plot(ax = ax,column = 'gap',edgecolor = (0,0,0,0),cmap = cmap,vmax = vmax,vmin = -vmax)
# 不显示坐标轴
plt.axis('off')    

# 绘制colorbar
plt.imshow([[-vmax,vmax]], cmap=cmap)
cax = plt.axes([0.13, 0.32, 0.02, 0.3])
plt.colorbar(cax=cax)
ax.set_xlim(113.6,114.8)
ax.set_ylim(22.4,22.9)

plt.show()
fig.savefig('./images/gap-gis.png')

# 结论：
# 如图，红色区域表示高收入更倾向于接客的地点，蓝色区域表示中等收入群体更倾向于接客的地点  
# 市中心一片红，郊区一片蓝，高收入车手更喜欢在市中心接客，尤其在罗湖福田中心区，
# 且更多出现在一些枢纽区域（保安机场，罗湖火车站，皇岗口岸，深圳湾口岸等等）  

# TODO 问题七：高收入与中收入者OD的差别
# 栅格化代码
# 划定栅格划分范围
lon1 = 113.75194
lon2 = 114.624187
lat1 = 22.447837
lat2 = 22.864748
latStart = min(lat1, lat2);
lonStart = min(lon1, lon2);

# 定义栅格大小(单位m)
accuracy = 2000;
# 计算栅格的经纬度增加量大小▲Lon和▲Lat
deltaLon = accuracy * 360 / (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360));
deltaLat = accuracy * 360 / (2 * math.pi * 6371004);

# 高收入OD统计
highincome_order['SLONCOL'] = ((highincome_order['Lng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
highincome_order['SLATCOL'] = ((highincome_order['Lat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')
highincome_order['ELONCOL'] = ((highincome_order['ELng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
highincome_order['ELATCOL'] = ((highincome_order['ELat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')
hod = highincome_order[['SLONCOL','SLATCOL','ELONCOL','ELATCOL']].copy()
hod['highcount'] = 1
hod = hod.groupby(['SLONCOL','SLATCOL','ELONCOL','ELATCOL'])['highcount'].count().reset_index()

# 中收入OD统计
midincome_order['SLONCOL'] = ((midincome_order['Lng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
midincome_order['SLATCOL'] = ((midincome_order['Lat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')
midincome_order['ELONCOL'] = ((midincome_order['ELng'] - (lonStart - deltaLon / 2))/deltaLon).astype('int')
midincome_order['ELATCOL'] = ((midincome_order['ELat'] - (latStart - deltaLat / 2))/deltaLat).astype('int')
mod = midincome_order[['SLONCOL','SLATCOL','ELONCOL','ELATCOL']].copy()
mod['midcount'] = 1
mod = mod.groupby(['SLONCOL','SLATCOL','ELONCOL','ELATCOL'])['midcount'].count().reset_index()

# 高收入中收入OD合计
allod = pd.merge(hod,mod,on = ['SLONCOL','SLATCOL','ELONCOL','ELATCOL'],how = 'outer').fillna(0)
allod['count'] = allod['highcount'] - allod['midcount']
print(allod.head(5))

#计算起点栅格的中心点经纬度
allod['SHBLON'] = allod['SLONCOL'] * deltaLon + (lonStart - deltaLon / 2)
allod['SHBLAT'] = allod['SLATCOL'] * deltaLat + (latStart - deltaLat / 2)
#计算终点栅格的中心点经纬度
allod['EHBLON'] = allod['ELONCOL'] * deltaLon + (lonStart - deltaLon / 2)
allod['EHBLAT'] = allod['ELATCOL'] * deltaLat + (latStart - deltaLat / 2)

#按从小到大排序，并且计算alpha值
vmax = max(allod['count'].max(),-allod['count'].min())
allod['alpha'] = abs(allod['count'])/vmax
allod = allod.sort_values(by = 'alpha')
print(vmax)

fig = plt.figure(1,(10,8),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
bounds = [113.6,22.4,114.8,22.9]

# 绘制底图的农村网太慢了，暂时不画底图，直接绘制行政区划
shp = r'shapefile/sz.shp'
xzqh = geopandas.GeoDataFrame.from_file(shp,encoding = 'utf-8')
xzqh.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0.2),linewidths=0.5)

# 设定colormap
import matplotlib as mpl
vmax = max(allod['count'].max(),-allod['count'].min())
norm = mpl.colors.Normalize(vmin=-vmax,vmax=vmax)
cmapname = 'seismic'
cmap = matplotlib.cm.get_cmap(cmapname)

# 绘制OD
print('绘制OD')
import time
starttime = time.time()
for i in range(len(allod)):
    # 设定plt.plot里面的参数alpha和color和linewidth
    color_i=cmap(norm(allod['count'].iloc[i]))
    linewidth_i=3*max(0.05,allod['alpha'].iloc[i])
    plt.plot([allod['SHBLON'].iloc[i],allod['EHBLON'].iloc[i]],
             [allod['SHBLAT'].iloc[i],allod['EHBLAT'].iloc[i]],
             color=color_i,linewidth=linewidth_i)
endtime = time.time()
print('OD绘制时间：',endtime-starttime)
    
plt.axis('off')
plt.imshow([[-vmax,vmax]], cmap=cmap)
cax = plt.axes([0.13, 0.32, 0.02, 0.3])
plt.colorbar(cax=cax)

ax.set_xlim(113.6,114.8)
ax.set_ylim(22.4,22.9)
plt.show()
fig.savefig('./images/od-gap.png')

# 更快速度的办法-用geopandas来画图
from shapely.geometry import LineString
#生成GeoDataFrame
odgpd = geopandas.GeoDataFrame()
odgpd['alpha'] = allod['alpha']
odgpd['count'] = allod['count']
odgpd['geometry'] = allod.apply(lambda r:LineString([[r['SHBLON'],r['SHBLAT']],[r['EHBLON'],r['EHBLAT']]]),axis = 1)
#将最小的线粗细定义为0.05
odgpd.loc[odgpd['alpha']<0.05,'alpha'] = 0.05

fig = plt.figure(1,(10,8),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
bounds = [113.6,22.4,114.8,22.9]

shp = r'shapefile/sz.shp'
xzqh = geopandas.GeoDataFrame.from_file(shp,encoding = 'utf-8')
xzqh.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0.2),linewidths=0.5)

vmax = max(allod['count'].max(),-allod['count'].min())
norm = mpl.colors.Normalize(vmin=-vmax,vmax=vmax)
cmapname = 'seismic'
cmap = matplotlib.cm.get_cmap(cmapname)

print('绘制OD2')
import time
starttime = time.time()
odgpd.plot(ax = ax,column = 'count',vmax = vmax,vmin = -vmax,cmap = cmap,linewidth = 3*odgpd['alpha'])
endtime = time.time()
print('OD2绘制时间：',endtime-starttime)
    
plt.axis('off')    
plt.imshow([[-vmax,vmax]], cmap=cmap)
cax = plt.axes([0.13, 0.32, 0.02, 0.3])
plt.colorbar(cax=cax)
ax.set_xlim(113.6,114.8)
ax.set_ylim(22.4,22.9)
plt.show()
fig.savefig('./images/od-gap-2.png')

# TODO 问题4-7：高收入车手策略总结
# 从上面的分析来看，高收入群体确实存在高收入的秘诀，总结一下，那就是：  

# 1.勤奋，你需要比别人更勤奋，每天跑更多的路程，接更多的单  
# 2.选择性接单（拒载），你需要比别人接更多的短距离出行订单（高收入者接的订单平均出行距离比中收入者短！）  
# 3.工作时间，高收入者在夜间、凌晨的订单比例比中收入者更高，半夜开车挣钱多！  
# 4.技术，你需要懂得怎么挣钱，在控制自己的空载行驶路程与别人持平的同时，增加自己载客的行驶路程  
# 5.接客地点，尽量在市中心，单价会更高

# TODO 问题八：进一步分析：空载时候的候车时间有没有差别？
# 猜测一个出租车手的技术体现(收入差距)不在于载客状态，而在于空载状态
# 为什么高收入出租车手每天的订单数量多，而每天的空载里程却跟中等收入的选手差距不大？

data = datapro # 读取数据
intervaldata = data[data['distance']==0]['interval']/3600

# 用pandas自带hist绘制直方图
import matplotlib.pyplot as plt
fig = plt.figure(1,(10,6),dpi = 100)    
ax1 = plt.subplot(211)
plt.sca(ax1)
intervaldata.hist(ax = ax1,bins = 400)
plt.ylabel('Count')
plt.xlabel('Stop time (Hour)')
plt.xticks(range(24),range(24))
plt.title('Histogram of Stop time')
plt.xlim(0,5)

ax1 = plt.subplot(212)
plt.sca(ax1)
(intervaldata[intervaldata<1]*60).hist(ax = ax1,bins = 400)
plt.ylabel('Count')
plt.xlabel('Stop time (minutes)')
plt.xticks(range(0,60,5),range(0,60,5))
plt.xlim(0,60)
plt.show()
fig.savefig('./images/stop-time.png')

# 停车时间绝大部分都是非常短的，剔除30分钟以上的停车摸鱼时间
# 删掉30分钟以上的停车
data = data[-((data['distance']==0)&(data['interval']>=1800))]

# 高收入数据和中收入数据打上标签后合并在一起处理
highincome_data = pd.merge(data,highincome,on = 'VehicleNum')
highincome_data['Index'] = 'High income'
highincome_data['isstop'] = highincome_data['distance']==0
midincome_data = pd.merge(data,midincome,on = 'VehicleNum')
midincome_data['Index'] = 'Middle income'
midincome_data['isstop'] = midincome_data['distance']==0

# 计算停车时间占运营时间的比例
tmp1 = pd.concat([midincome_data,highincome_data]).groupby(['isstop','VehicleNum','OpenStatus','Index'])['interval'].sum().reset_index()
tmp1 = pd.merge(tmp1[tmp1['isstop']],tmp1[-tmp1['isstop']],on = ['VehicleNum','OpenStatus','Index'])
tmp1['rate'] = tmp1['interval_x']/(tmp1['interval_x']+tmp1['interval_y'])

# 绘图
fig = plt.figure(1,(6,2),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
sns.boxplot(x="OpenStatus", y="rate",hue="Index", data=tmp1,hue_order=['High income','Middle income']
,ax = ax)
plt.ylabel('Stop time proportion')
plt.xticks([0,1],['Idle','Delivery'])
plt.ylim(0,0.8)
plt.show()
fig.savefig('./images/stop-time-proportion.png')

# 数据显示的与先前的猜测正好相反，高收入在空载时和载客时的停车时间比例都要比中等收入的高，原因居然是
# 测试结果为：空载时多停车，减少油耗；载客时反而要多停车，增加候时费……