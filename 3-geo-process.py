# 使用python的geopandas包创建栅格，对出租车OD数据进行栅格对应，并绘制专题图

# 初始数据为深圳行政区划的GIS文件和出租车OD数据

# 导入必要的包
import pandas as pd
import numpy as np

# 绘制图用的包
import matplotlib as mpl
import matplotlib.pyplot as plt

# geopandas包
import geopandas

# shapely包
from shapely.geometry import Point, Polygon, shape

# 读取shapefile文件
shp = r"shapefile/sz.shp"
sz = geopandas.GeoDataFrame.from_file(shp, encoding="utf-8")

# 栅格化代码
import math

# 定义一个测试栅格划的经纬度
testlon = 114
testlat = 22.5

# 划定栅格划分范围
lon1 = 113.75194
lon2 = 114.624187
lat1 = 22.447837
lat2 = 22.864748

latStart = min(lat1, lat2)
lonStart = min(lon1, lon2)

# 定义栅格大小(单位m)
accuracy = 500

# 计算栅格的经纬度增加量大小▲Lon和▲Lat
deltaLon = (
    accuracy * 360 / (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360))
)
deltaLat = accuracy * 360 / (2 * math.pi * 6371004)

# 计算栅格的经纬度编号
LONCOL = divmod(float(testlon) - (lonStart - deltaLon / 2), deltaLon)[0]
LATCOL = divmod(float(testlat) - (latStart - deltaLat / 2), deltaLat)[0]

# 计算栅格的中心点经纬度
HBLON = LONCOL * deltaLon + (lonStart - deltaLon / 2)  # 格子编号*格子宽+起始横坐标-半个格子宽=格子中心横坐标
HBLAT = LATCOL * deltaLat + (latStart - deltaLat / 2)

# 把算好的东西print出来看看
LONCOL, LATCOL, HBLON, HBLAT, deltaLon, deltaLat

# 另外，我们要生成这些栅格的geopandas数据
from shapely.geometry import Point, Polygon, shape

Polygon(
    [
        (HBLON + deltaLon / 2, HBLAT - deltaLat / 2),
        (HBLON + deltaLon / 2, HBLAT + deltaLat / 2),
        (HBLON - deltaLon / 2, HBLAT + deltaLat / 2),
        (HBLON - deltaLon / 2, HBLAT - deltaLat / 2),
    ]
)

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Point, Polygon, shape


# 定义空的geopandas表
data = geopandas.GeoDataFrame()

# 定义空的list，后面循环一次就往里面加东西
LONCOL = []
LATCOL = []
geometry = []
HBLON1 = []
HBLAT1 = []

# 计算总共要生成多少个栅格
# lon方向是lonsnum个栅格
lonsnum = int((lon2 - lon1) / deltaLon) + 1
# lat方向是latsnum个栅格
latsnum = int((lat2 - lat1) / deltaLat) + 1

for i in range(lonsnum):
    for j in range(latsnum):
        HBLON = i * deltaLon + (lonStart - deltaLon / 2)
        HBLAT = j * deltaLat + (latStart - deltaLat / 2)
        # 把生成的数据都加入到前面定义的空list里面
        LONCOL.append(i)
        LATCOL.append(j)
        HBLON1.append(HBLON)
        HBLAT1.append(HBLAT)

        # 生成栅格的Polygon形状
        # 这里我们用周围的栅格推算三个顶点的位置，否则生成的栅格因为小数点取值的问题会出现小缝，无法完美覆盖
        HBLON_1 = (i + 1) * deltaLon + (lonStart - deltaLon / 2)
        HBLAT_1 = (j + 1) * deltaLat + (latStart - deltaLat / 2)
        geometry.append(
            Polygon(
                [
                    (HBLON - deltaLon / 2, HBLAT - deltaLat / 2),
                    (HBLON_1 - deltaLon / 2, HBLAT - deltaLat / 2),
                    (HBLON_1 - deltaLon / 2, HBLAT_1 - deltaLat / 2),
                    (HBLON - deltaLon / 2, HBLAT_1 - deltaLat / 2),
                ]
            )
        )

# 为geopandas文件的每一列赋值为刚刚的list
data["LONCOL"] = LONCOL
data["LATCOL"] = LATCOL
data["HBLON"] = HBLON1
data["HBLAT"] = HBLAT1
data["geometry"] = geometry


# 取栅格和深圳行政区划的交集栅格
grid = data[data.intersects(sz.unary_union)]
grid.plot()

# 保存
grid.to_file(r"shapefile\grid", encoding="utf-8")

# 将数据对应到栅格（这里不用低效率的循环）
import pandas as pd

TaxiOD = pd.read_csv(r"data-sample/TaxiOD.csv")
TaxiOD.columns = ["VehicleNum", "Stime", "SLng", "SLat", "ELng", "ELat", "Etime"]

# 计算栅格的经纬度编号、中心点经纬度
TaxiOD = TaxiOD[-TaxiOD["ELng"].isnull()].copy()
TaxiOD["SLONCOL"] = ((TaxiOD["SLng"] - (lonStart - deltaLon / 2)) / deltaLon).astype(
    "int"
)
TaxiOD["SLATCOL"] = ((TaxiOD["SLat"] - (latStart - deltaLat / 2)) / deltaLat).astype(
    "int"
)
TaxiOD["SHBLON"] = TaxiOD["SLONCOL"] * deltaLon + (lonStart - deltaLon / 2)
TaxiOD["SHBLAT"] = TaxiOD["SLATCOL"] * deltaLat + (latStart - deltaLat / 2)
TaxiOD["ELONCOL"] = ((TaxiOD["ELng"] - (lonStart - deltaLon / 2)) / deltaLon).astype(
    "int"
)
TaxiOD["ELATCOL"] = ((TaxiOD["ELat"] - (latStart - deltaLat / 2)) / deltaLat).astype(
    "int"
)
TaxiOD["EHBLON"] = TaxiOD["ELONCOL"] * deltaLon + (lonStart - deltaLon / 2)
TaxiOD["EHBLAT"] = TaxiOD["ELATCOL"] * deltaLat + (latStart - deltaLat / 2)
# 筛选去掉起点终点在同一个格子里的OD
# 即筛选去掉不在研究范围内的栅格，TaxiOD的LONCOL、LATCOL都需要在我们的范围内
TaxiOD = TaxiOD[
    -(
        (TaxiOD["SLONCOL"] == TaxiOD["ELONCOL"])
        & (TaxiOD["SLATCOL"] == TaxiOD["ELATCOL"])
    )
]
# 筛选去掉不在研究范围内的栅格
TaxiOD = TaxiOD[
    (TaxiOD["SLONCOL"] >= 0)
    & (TaxiOD["SLATCOL"] >= 0)
    & (TaxiOD["ELONCOL"] >= 0)
    & (TaxiOD["ELATCOL"] >= 0)
    & (TaxiOD["SLONCOL"] <= lonsnum)
    & (TaxiOD["SLATCOL"] <= latsnum)
    & (TaxiOD["ELONCOL"] <= lonsnum)
    & (TaxiOD["ELATCOL"] <= latsnum)
]
TaxiOD.head(5)

# 集计栅格OD（全天、高峰时段）
OD = (
    TaxiOD.groupby(["SLONCOL", "SLATCOL", "ELONCOL", "ELATCOL"])["VehicleNum"]
    .count()
    .reset_index()
)
# OD按照大小排列
OD = OD.sort_values(by="VehicleNum", ascending=False)

# 绘制栅格的OD矩阵图

# 取前20的OD
Topod = OD.iloc[:20].copy()

# 计算起点栅格的中心点经纬度
Topod["SHBLON"] = Topod["SLONCOL"] * deltaLon + (lonStart - deltaLon / 2)
Topod["SHBLAT"] = Topod["SLATCOL"] * deltaLat + (latStart - deltaLat / 2)

# 计算终点栅格的中心点经纬度
Topod["EHBLON"] = Topod["ELONCOL"] * deltaLon + (lonStart - deltaLon / 2)
Topod["EHBLAT"] = Topod["ELATCOL"] * deltaLat + (latStart - deltaLat / 2)

# 导入绘图包
import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure(1, (10, 8), dpi=250)
ax = plt.subplot(111)
plt.sca(ax)

# 把刚才生成的栅格在ax上绘制
grid.plot(ax=ax, edgecolor=(0, 0, 0, 0.8), facecolor=(0, 0, 0, 0), linewidths=0.2)

# 把合并的行政区划变成一个geopandas，在ax上绘制
SZ_all = geopandas.GeoDataFrame()
SZ_all["geometry"] = [sz.unary_union]
SZ_all.plot(ax=ax, edgecolor=(0, 0, 0, 1), facecolor=(0, 0, 0, 0), linewidths=0.5)

plt.show()
fig.savefig("./images/SZ.png")


fig = plt.figure(1, (10, 8), dpi=250)
ax = plt.subplot(111)
plt.sca(ax)

grid.plot(ax=ax, edgecolor=(0, 0, 0, 0.8), facecolor=(0, 0, 0, 0), linewidths=0.2)
SZ_all.plot(ax=ax, edgecolor=(0, 0, 0, 1), facecolor=(0, 0, 0, 0), linewidths=0.5)

for i in range(len(Topod)):
    plt.plot(
        [Topod["SHBLON"].iloc[i], Topod["EHBLON"].iloc[i]],
        [Topod["SHBLAT"].iloc[i], Topod["EHBLAT"].iloc[i]],
    )

# 不显示坐标轴
plt.axis("off")
plt.show()
fig.savefig("./images/Top20OD.png")

# 绘制全部的OD
OD1 = OD[OD["VehicleNum"] > 10].copy()

# OD从小到大排序方便我们后续操作，因为我们希望小的OD先画，放在最底下，大的OD后画，放在最上面
OD1 = OD1.sort_values(by="VehicleNum")

# 计算起点栅格的中心点经纬度
OD1["SHBLON"] = OD1["SLONCOL"] * deltaLon + (lonStart - deltaLon / 2)
OD1["SHBLAT"] = OD1["SLATCOL"] * deltaLat + (latStart - deltaLat / 2)

# 计算终点栅格的中心点经纬度
OD1["EHBLON"] = OD1["ELONCOL"] * deltaLon + (lonStart - deltaLon / 2)
OD1["EHBLAT"] = OD1["ELATCOL"] * deltaLat + (latStart - deltaLat / 2)

# 对OD分5组，生成一个取值为0-1的列，每组的值相同，用以表示OD的粗细，取名linewidth
step = 5
OD1["linewidth"] = (np.array(range(len(OD1))) * step / len(OD1)).astype(
    "int"
) / step + 0.1

# 绘制
# 如果遍历绘制OD，绘制速度比较慢，绘制5319条OD用时31s。  
# 但是，如果把DataFrame变成GeoDataFrame，然后用自带的plot函数绘制，会快很多
from shapely.geometry import LineString
OD1['geometry'] = OD1.apply(lambda r:LineString([[r['SHBLON'],r['SHBLAT']],[r['EHBLON'],r['EHBLAT']]]),axis = 1)
OD1 = geopandas.GeoDataFrame(OD1)

fig = plt.figure(1,(10,8),dpi = 250)    
ax = plt.subplot(111)
plt.sca(ax)

#计时
import time
timeflag = time.time()
#绘制底图
grid.plot(ax = ax,edgecolor = (0,0,0,0.8),facecolor = (0,0,0,0),linewidths=0.2)
SZ_all.plot(ax = ax,edgecolor = (0,0,0,1),facecolor = (0,0,0,0),linewidths=0.5)
print('绘制底图用时',time.time()-timeflag,'秒')


#设置colormap的数据
import matplotlib
vmax = OD['VehicleNum'].max()
cmapname = 'autumn_r'
cmap = matplotlib.cm.get_cmap(cmapname)

timeflag = time.time()
#绘制OD
OD1.plot(ax = ax,column = 'VehicleNum',vmax = vmax,vmin = 0,cmap = cmap,linewidth = OD1['linewidth'])
print('绘制OD用时',time.time()-timeflag,'秒')

plt.axis('off')    
plt.imshow([[0,vmax]], cmap=cmap)
cax = plt.axes([0.08, 0.4, 0.02, 0.3])
plt.colorbar(cax=cax)
ax.set_xlim(113.6,114.8)
ax.set_ylim(22.4,22.9)

plt.show()
fig.savefig('./images/ODDraw.png',dpi = 250)
