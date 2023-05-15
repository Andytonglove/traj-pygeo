import pandas as pd

# 一、出租车GPS数据清洗

# 读取数据
data = pd.read_csv(r"data-sample/TaxiData-Sample", header=None)
# 给数据命名列
data.columns = ["VehicleNum", "Stime", "Lng", "Lat", "OpenStatus", "Speed"]

# 将数据排序,并把排序后的数据赋值给原来的数据
data = data.sort_values(by=["VehicleNum", "Stime"])
data.head(5)

# 数据格式：
# >VehicleNum —— 车牌  
# Stime —— 时间  
# Lng —— 经度  
# Lat —— 纬度  
# OpenStatus —— 是否有乘客(0没乘客，1有乘客)  
# Speed —— 速度  

# 清洗异常数据
data = data[
    -(
        (data["OpenStatus"].shift(-1) == data["OpenStatus"].shift())
        & (data["OpenStatus"].shift(-1) != data["OpenStatus"])
        & (data["VehicleNum"].shift(-1) == data["VehicleNum"].shift())
        & (data["VehicleNum"].shift(-1) == data["VehicleNum"])
    )
]

# 识别OD点
# 让这几个字段的下一条数据赋值给新的字段，在字段名加个1，代表后面一条数据的值
data.loc[:, "OpenStatus1"] = data["OpenStatus"].shift(-1)
data.loc[:, "VehicleNum1"] = data["VehicleNum"].shift(-1)
data.loc[:, "Lng1"] = data["Lng"].shift(-1)
data.loc[:, "Lat1"] = data["Lat"].shift(-1)
data.loc[:, "Stime1"] = data["Stime"].shift(-1)


data.loc[:, "StatusChange"] = data["OpenStatus1"] - data["OpenStatus"]

# 将上下车状态整理为OD
data = data[
    ((data["StatusChange"] == 1) | (data["StatusChange"] == -1))
    & (data["VehicleNum"] == data["VehicleNum1"])
]

# data数据只保留一些我们需要的字段
data = data[["VehicleNum", "Stime", "Lng", "Lat", "StatusChange"]]
data.head(5)

# 保存数据
data = data.rename(columns={"Lng": "SLng", "Lat": "SLat"})
data["ELng"] = data["SLng"].shift(-1)
data["ELat"] = data["SLat"].shift(-1)
data["Etime"] = data["Stime"].shift(-1)
data = data[data["StatusChange"] == 1]
data = data.drop("StatusChange", axis=1)

data.to_csv(r"data-sample/TaxiOD-Sample1.csv", index=None)
