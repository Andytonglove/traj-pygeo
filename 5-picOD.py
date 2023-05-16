# 绘制数据的分布情况：即散点图和热力图
# 出租车原始GPS数据(在data-sample文件夹下，原始数据集的抽样500辆车的数据)

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_map
import seaborn as sns
import numpy as np

# 数据集计处理
# 读取数据
data = pd.read_csv(r'data-sample\TaxiData-Sample',header = None)
# 给数据命名列
data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat', 'OpenStatus', 'Speed']

# 筛选范围内数据
bounds = [113.7, 22.42, 114.3, 22.8]
data = data[(data['Lng']>bounds[0])&(data['Lng']<bounds[2])&(data['Lat']>bounds[1])&(data['Lat']<bounds[3])]

# 经纬度小数点保留三位小数
data2 = data[['Lng','Lat']].round(3).copy()

# 集计每个小范围内数据量
data2['count'] = 1
data2 = data2.groupby(['Lng','Lat'])['count'].count().reset_index()

# 排序数据，让数据量小的放上面先画，数据大的放下面最后画
data2.sort_values(by = 'count')


# 散点图绘制
bounds = [113.7, 22.42, 114.3, 22.8]

fig = plt.figure(1,(8,8),dpi = 100)    
ax = plt.subplot(111)
plt.sca(ax)
fig.tight_layout(rect = (0.05,0.1,1,0.9))

#背景
plot_map.plot_map(plt,bounds,zoom = 12,style = 4)

# colorbar
pallete_name = "BuPu"
colors = sns.color_palette(pallete_name, 3)
colors.reverse()
cmap = mpl.colors.LinearSegmentedColormap.from_list(pallete_name, colors)
vmax = data2['count'].quantile(0.99)
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

# plot scatters
plt.scatter(data2['Lng'],data2['Lat'],s = 1,alpha = 1,c = data2['count'],cmap = cmap,norm=norm )
plt.axis('off')
plt.xlim(bounds[0],bounds[2])
plt.ylim(bounds[1],bounds[3])

# 加比例尺和指北针
plot_map.plotscale(ax,bounds = bounds,textsize = 10,compasssize = 1,accuracy = 2000,rect = [0.06,0.03])

# 假colorbar
plt.imshow([[0,vmax]], cmap=cmap)
cax = plt.axes([0.13, 0.33, 0.02, 0.3])
plt.colorbar(cax=cax)

plt.show()
fig.savefig('./images/scatter.png')

# 热力图绘制(使用contourf)
import numpy as np

d = data2.pivot(columns = 'Lng',index = 'Lat',values = 'count').fillna(0)
z = np.log(d.values)
x = d.columns
y = d.index

levels = np.linspace(0, z.max(), 25)
bounds = [113.7, 22.42, 114.3, 22.8]

#   -- plot --
fig = plt.figure(1,(10,10),dpi = 60)  
ax = plt.subplot(111)
plt.sca(ax)
fig.tight_layout(rect = (0.05,0.1,1,0.9))#调整整体空白

# 绘制底图
plot_map.plot_map(plt,bounds,zoom = 12,style = 4)

# colorbar的数据
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#9DCC42','#FFFE03','#F7941D','#E9420E','#FF0000'], 256)

# 绘制热力图
plt.contourf(x,y,z, levels=levels, cmap=cmap,origin = 'lower')
plt.axis('off')
plt.xlim(bounds[0],bounds[2])
plt.ylim(bounds[1],bounds[3])

# 绘制假的colorbar
plt.imshow([np.exp(levels)], cmap=cmap)
cax = plt.axes([0.13, 0.32, 0.02, 0.3])
plt.colorbar(cax=cax)

plt.show()
fig.savefig('./images/heatmap1.png')


# 热力图绘制(还可采用seaborn-kdeplot)
# 热力图绘制（scipy）
import scipy
scipy.__version__
def heatmapplot(data,weight,gridsize = 100,bw = 'scott',cmap = plt.cm.gist_earth_r, ax=None,**kwargs):
    #数据整理
    from scipy import stats
    m1 = data[:,0]
    m2 = data[:,1]
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    X, Y = np.mgrid[xmin:xmax:(xmax-xmin)/gridsize, ymin:ymax:(ymax-ymin)/gridsize]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    #用scipy计算带权重的高斯kde
    kernel = stats.gaussian_kde(values,bw_method = bw,weights = weight)
    Z = np.reshape(kernel(positions).T, X.shape)
    #绘制contourf
    cset = ax.contourf(Z.T,extent=[xmin, xmax, ymin, ymax],cmap = cmap,**kwargs)
    #设置最底层为透明
    cset.collections[0].set_alpha(0)
    
    return cset

bounds = [113.7, 22.42, 114.3, 22.8]

#   -- plot --
fig = plt.figure(1,(10,10),dpi = 60)    
ax = plt.subplot(111)
plt.sca(ax)
fig.tight_layout(rect = (0.05,0.1,1,0.9))#调整整体空白

#绘制底图
plot_map.plot_map(plt,bounds,zoom = 12,style = 4)

#colorbar的数据
import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#9DCC42','#FFFE03','#F7941D','#E9420E','#FF0000'], 256)

#设定位置
plt.axis('off')
plt.xlim(bounds[0],bounds[2])
plt.ylim(bounds[1],bounds[3])

#绘制热力图
cset = heatmapplot(data2.values,  #输入经纬度数据
            data2['count'],       #输入每个点的权重
            alpha = 0.8,          #透明度
            gridsize = 80,        #绘图精细度，越高越慢
            bw = 0.03,            #高斯核大小（经纬度），越小越精细
            cmap = cmap,
            ax = ax
           )

#定义colorbar位置
cax = plt.axes([0.13, 0.32, 0.02, 0.3])
plt.colorbar(cset,cax=cax)

plt.show()
fig.savefig('./images/heatmap2.png')