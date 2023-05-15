# 使用python的matplotlib包和seaborn包对出租车GPS数据绘制数据统计图表

# 二、数据统计图表绘制
import pandas as pd

# 根据出租车GPS数据和OD数据，绘制出租车GPS数据的统计图表
data = pd.read_csv(r"data-sample/TaxiData-Sample", header=None)
data.columns = ["VehicleNum", "Stime", "Lng", "Lat", "OpenStatus", "Speed"]

# 读取OD数据
TaxiOD = pd.read_csv(r"data-sample/TaxiOD.csv")
TaxiOD.columns = ["VehicleNum", "Stime", "SLng", "SLat", "ELng", "ELat", "Etime"]

# 每小时GPS数据量绘图

# 计时
import time

timeflag = time.time()
# 方法1：把时间当成字符串，用列自带的str方法，取前两位
data["Hour"] = data["Stime"].str.slice(0, 2)
# 计算耗时
print("方法1", time.time() - timeflag, "s")
timeflag = time.time()

# 方法2：把时间当成字符串，遍历取字符串前两位
data["Hour"] = data["Stime"].apply(lambda r: r[:2])
# 计算耗时
print("方法2", time.time() - timeflag, "s")
timeflag = time.time()

# 方法3：转换为时间格式，后提取小时（非常慢）
data["Hour"] = pd.to_datetime(data["Stime"]).apply(lambda r: r.hour)
# 计算耗时
print("方法3", time.time() - timeflag, "s")
timeflag = time.time()

# 这个是对每一列都计数了，所以取其中一列出来，例如我这里取了['VehicleNum']
hourcount = (
    data.groupby(data["Stime"].apply(lambda r: r[:2]))["VehicleNum"]
    .count()
    .reset_index()
)

# 用matplotlib包来绘制每小时GPS数据量的图表
import matplotlib.pyplot as plt
fig = plt.figure(1,(8,4),dpi = 250)    
ax = plt.subplot(111)
plt.sca(ax)

#折线图调整颜色加上数据点
plt.plot(hourcount['Stime'],hourcount['VehicleNum'],'k-',hourcount['Stime'],hourcount['VehicleNum'],'k.')
#加上条形图
plt.bar(hourcount['Stime'],hourcount['VehicleNum'],width =0.5)

plt.title('Hourly data Volume')

#把y轴起点固定在0
plt.ylim(0,80000)
plt.ylabel('Data volumn')
plt.xlabel('Hour')
plt.show()

# 加上seaborn的主题
import seaborn as sns

sns.set_style("darkgrid", {"xtick.major.size": 10, "ytick.major.size": 10})

import matplotlib.pyplot as plt

fig = plt.figure(1, (8, 4), dpi=250)
ax = plt.subplot(111)
plt.sca(ax)

plt.plot(
    hourcount["Stime"],
    hourcount["VehicleNum"],
    "k-",
    hourcount["Stime"],
    hourcount["VehicleNum"],
    "k.",
)
plt.bar(hourcount["Stime"], hourcount["VehicleNum"], width=0.5)

plt.title("Hourly data Volume")

plt.ylim(0, 80000)
plt.ylabel("Data Volume")
plt.xlabel("Hour")
plt.show()

# 对TaxiOD数据绘制OD数据量的图表，绘制订单持续时间分布图

TaxiOD = TaxiOD[-TaxiOD["Etime"].isnull()]

timeflag = time.time()
# 方法1：直接硬算
TaxiOD["order_time"] = (
    TaxiOD["Etime"].str.slice(0, 2).astype("int") * 3600
    + TaxiOD["Etime"].str.slice(3, 5).astype("int") * 60
    + TaxiOD["Etime"].str.slice(6, 8).astype("int")
    - TaxiOD["Stime"].str.slice(0, 2).astype("int") * 3600
    - TaxiOD["Stime"].str.slice(3, 5).astype("int") * 60
    - TaxiOD["Stime"].str.slice(6, 8).astype("int")
)

# 计算耗时
print("方法1", time.time() - timeflag, "s")
timeflag = time.time()

# 方法2：转换为时间格式，相减后提取秒（非常慢）
TaxiOD["order_time"] = pd.to_datetime(TaxiOD["Etime"]) - pd.to_datetime(TaxiOD["Stime"])
TaxiOD["order_time"] = TaxiOD["order_time"].apply(lambda r: r.seconds)

# 计算耗时
print("方法2", time.time() - timeflag, "s")
timeflag = time.time()

fig = plt.figure(1,(10,5),dpi = 250)     
ax = plt.subplot(111)
plt.sca(ax)

#整理数据
hour = TaxiOD['Hour'].drop_duplicates().sort_values()
datas = []
for i in range(len(hour)):
    datas.append(TaxiOD[TaxiOD['Hour']==hour.iloc[i]]['order_time']/60)
#绘制
plt.boxplot(datas)
#更改x轴ticks的文字
plt.xticks(range(1,len(hour)+1),list(hour))

plt.ylabel('Order time(minutes)')
plt.xlabel('Order start time')
plt.ylim(0,60)


plt.show()


# 用seaborn包绘制以每小时分组的订单时间分布，只需要输入整个数据，就可以很方便的画出来

fig = plt.figure(1,(10,5),dpi = 250)    
ax = plt.subplot(111)
plt.sca(ax)

#只需要一行
sns.boxplot(x="Hour", y=TaxiOD["order_time"]/60, data=TaxiOD,ax = ax)

plt.ylabel('Order time(minutes)')
plt.xlabel('Order start time')
plt.ylim(0,60)
plt.show()