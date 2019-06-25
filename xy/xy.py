import sys
import pymysql
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from LeuvenMapMatching.leuvenmapmatching.map.inmem import InMemMap
from LeuvenMapMatching.leuvenmapmatching.matcher.distance import DistanceMatcher
import threading
import time
from LeuvenMapMatching.leuvenmapmatching import visualization as mmviz
import osmread
from sklearn.cluster import DBSCAN
from  decimal import Decimal

import osmnx as ox
import hdbscan
import matplotlib.pyplot as plt
import os

from LeuvenMapMatching.leuvenmapmatching.matcher import simple as simpleMatching

import pickle


# geo数据存储
from sqlalchemy import create_engine
def convertFun(m,n):
    splited_str = m + ' ' + n
    import time
    timeArray = time.strptime(splited_str, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def write_data2sql():
    engine = create_engine('mysql+pymysql://root:123456@localhost:3306/test')
    # 得到路径list
    path='./geoData/'
    os.listdir(path)
    listDir = [path+ i + '/Trajectory/' for i in os.listdir(path)]
    allDir = []
    for i in listDir:
        allDir.extend([i + j for j in os.listdir(i)])
    # 读取每个list的数据并存入geoInfo (上线1000)
    count = 0
    for index,item in enumerate(allDir):
        if count == 1000:
            break
        # 只读取4列
        data = pd.DataFrame(np.loadtxt(item, skiprows=6, delimiter=',',dtype=str,usecols=(0, 1,5,6)))
        if len(data) >= 500:
            data['Time'] = data.apply(lambda row: convertFun(row[2],row[3]),axis =1)
            data['Id'] = count
            data = data[['Id',0,1,'Time']].astype('double')
            data.to_sql('geoInfo', con=engine,index= False,if_exists='append')
            count += 1
# write_data2sql()


# 数据库连接
def getTrajectory(table,id_num):
    # 打开数据库连接
    db = pymysql.connect("localhost", "root", "123456", "test", charset='utf8' )
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    sql = "SELECT * FROM {} where id = {}".format(table,id_num)
    try:
       # 执行SQL语句
        cursor.execute(sql)
       # 获取所有记录列表
        results = np.array(cursor.fetchall())
    except:
        print ("Error: unable to fetch data")
    # 关闭数据库连接
    db.close()
    return results


# 轨迹处理类
class trace_process:
    # 去噪声和停留点
    # input：TR，sleepTime，min_size
    # TR:[id,经度，纬度，时间]，sleepTime：停留时间，min_size：每个簇最小样本
    # return : TR (DataFrame 格式)
    def remove_noise_stop(self,TR,sleepTime= 10800,min_size=10):
        # HDBSCAN
#         clusterer = DBSCAN(eps=0.8, min_samples=5)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        # 时间->时间戳，参数：pandas DataFrame 格式
#         def convertDate(Date_time):
#             DateLiet = Date_time.values
#             timestamp = []
#             for item in DateLiet:
#                 Date_time = int(time.mktime(item.timetuple()))
#                 timestamp.append(Date_time)
#             return timestamp
        TR = pd.DataFrame(TR,columns=['Id','Lat','Lon','Time'])
#         print ('sss',TR)
#         TR['Time'] = TR[['Time']].apply(convertDate)
        print("原始大小",len(TR))
        clusterer.fit(np.array(TR)[:,1:3])
        TR['label'] = clusterer.labels_
         # 去除停留点
        stopCluser = []
        groups = TR.groupby('label')
        for clusterID, cluster in groups:
            if clusterID != -1 and (cluster['Time'].max()-cluster['Time'].min()) > sleepTime:
                stopCluser.append(clusterID)
        TR = TR[~TR['label'].isin(stopCluser)]
        print (set(clusterer.labels_))
        print ("去掉停留点",len(TR))
        # 去除新数据噪声
        clusterer.fit(np.array(TR)[:,1:3])
        TR['label'] = clusterer.labels_
        TR = TR[~TR['label'].isin([-1])]
        print ("去掉噪声",len(TR))
        return TR[TR.columns[:-1]]
    def show_trace_pig(self,TR):
        TR = pd.DataFrame(TR,columns=['Id','Lat','Lon','Time'])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(TR['Lon'],TR['Lat'])
        plt.show()


tp = trace_process()

TR = tp.remove_noise_stop(getTrajectory('geoInfo',5),sleepTime=3600)

# 得到路径与坐标box
def getPath_Box(TR):
    path =list( np.array(TR.apply(lambda x:(float(x[1]),float(x[2])),axis=1)))
    box = np.array(pd.DataFrame(TR)[['Lon','Lat']].apply(['min','max']))
    boxList = list(box[0])+list(box[1])
    return path, boxList

path,locationBox = getPath_Box(TR)

np.array(path).mean(axis=0)


# 使用osmnx

# # graph = ox.graph_from_place('beijing, china', network_type='drive')
# graph = ox.graph_from_point((40.00792367,116.29994668),distance=3000,network_type='drive_service')
# graph_proj = ox.project_graph(graph)
# G_project_file=open('G_project.pickle','wb')
# pickle.dump(graph,G_project_file)
# G_project_file.close()

# Create GeoDataFrames
file=open('G_project.pickle','rb')
graph_proj = pickle.load(file)
nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)


map_con = InMemMap("myosm", use_rtree=True, index_edges=True)

for nid, row in nodes_proj[['x', 'y']].iterrows():
    map_con.add_node(int(nid), (row['x'], row['y']))
for nid, row in edges_proj[['u', 'v']].iterrows():
    map_con.add_edge(row['u'], row['v'])
map_con.purge()


states, _ = matcher1.match(path)

nodes = matcher1.path_pred_onlynodes

print("States\n------")
print(states)

print("Nodes\n------")
print(nodes)

print("")
matcher1.print_lattice_stats()
