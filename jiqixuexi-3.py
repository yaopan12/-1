import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import datetime
import sklearn.preprocessing
import sklearn.cluster

air_data_path = './datasets/air_data.csv'
air_data = pd.read_csv(air_data_path) #读取数据
print(air_data.shape) #读取数据的规模

print("前5条数据为：")
print(air_data.head(5))#读取前5行数据
print('\n')

print("每列数据类型为：")
print(air_data.dtypes)
print('\n')

print("数据的基本统计信息为：")
print(air_data.describe().T)#用describe()函数表述数据的基本统计信息
print('\n')

#检查数据中是否有重复的会员ID
dup = air_data[air_data['MEMBER_NO'].duplicated()]
if len(dup) != 0:
    print("There are duplication in the data:")
    print(dup)

#统计数据集中缺失值情况。isnull 函数返回布尔值 DataFrame ，代表数据中的每一个元素是不是 None、NaN 等。
# any() 函数返回布尔值数组，每一个元素代表 DataFrame 的一列是否含有True
print("数据集是否有缺失数据：")
print(air_data.isnull().any())
print('\n')


#丢弃具有缺失值的样本，在DataFrame的一列数据上调用notnull()函数，返回布尔值数组，True代表不为空，False代表为空
boolean_filter = air_data['SUM_YR_1'].notnull() & air_data['SUM_YR_2'].notnull() \
                 & air_data['GENDER'].notnull() & air_data['WORK_CITY'].notnull() \
                 & air_data['WORK_PROVINCE'].notnull() \
                 & air_data['WORK_COUNTRY'].notnull() & air_data['AGE'].notnull()
air_data = air_data[boolean_filter]

filter_1 = air_data['SUM_YR_1'] != 0
filter_2 = air_data['SUM_YR_2'] != 0
filter_3 = air_data['GENDER'] != 0
filter_4 = air_data['WORK_CITY'] != 0
filter_5 = air_data['WORK_PROVINCE'] != 0
filter_6 = air_data['WORK_COUNTRY'] != 0
filter_7 = air_data['AGE'] != 0
air_data = air_data[filter_1 | filter_2 | filter_3 | filter_4 | filter_5 | filter_6 | filter_7]
print("丢弃后的数据规模为：")
print(air_data.shape)
print('\n')

'''LCRFM模型：
L = LOAD_TIME-FFP_DATE
R = LAST_TO_END
F = FLIGHT_COUNT
M = SEG_KM_SUM
C = avg_discount'''
load_time = datetime.datetime.strptime('2014/03/31', '%Y/%m/%d')
ffp_dates = [datetime.datetime.strptime(ffp_date, '%Y/%m/%d') for ffp_date in air_data['FFP_DATE']]
length_of_relationship = [(load_time - ffp_date).days for ffp_date in ffp_dates]
air_data['LEN_REL'] = length_of_relationship

features = ['LEN_REL','FLIGHT_COUNT','avg_discount','SEG_KM_SUM','LAST_TO_END']
data = air_data[features]
features = ['L','F','C','M','R']
data.columns = features

print("LCRFM模型下的前5行数据：")
print(data.head(5))
print(data.describe().T)
print('\n')


#对特征进行标准化，使得各特征的均值为 0、方差为 1。
ss = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)#标准化
data = ss.fit_transform(data) #数据转换
data = pd.DataFrame(data, columns=features)

data_db = data.copy()
print("标准化数据后的元数据：")
print(data.describe().T)
print('\n')

#K-means聚类算法
num_clusters = 5 #设置群体类别为5
km = sklearn.cluster.KMeans(n_clusters=num_clusters, n_jobs=4) #模型加载
km.fit(data) #模型训练

r1 = pd.Series(km.labels_).value_counts()
r2 = pd.DataFrame(km.cluster_centers_)
r = pd.concat([r2, r1], axis=1)
r.columns = list(data.columns) + ['counts']
print("5个群体的中心，及其样本数：")
print(r.describe().T)
print('\n')

print("LCRMF模型对每个样本预测的群体标签：")
print(km.labels_)
print('\n')

#RFM模型
data_rfm = data[['R','F','M']]
print("RFM模型下的前5行数据：")
print(data_rfm.head(5))
print('\n')

km.fit(data_rfm) #模型对只包含rfm数据集训练
print("RFM模型对每个样本预测的群体标签：")
print(km.labels_)
print('\n')

r1 = pd.Series(km.labels_).value_counts()
r2 = pd.DataFrame(km.cluster_centers_)
rr = pd.concat([r2, r1], axis=1)
rr = pd.DataFrame(ss.fit_transform(rr) )
rr.columns = list(data_rfm.columns) + ['counts']
print("3个群体的中心，及其样本数：")
print(rr.describe().T)
print('\n')


def radar_factory(num_vars, frame='circle'):
    # 计算得到evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # 使用1条线段连接指定点
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 旋转绘图，使第一个轴位于顶部
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """覆盖填充，以便默认情况下关闭该行"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """覆盖填充，以便默认情况下关闭该行"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: x[0], y[0] 处的标记加倍
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # 轴必须以（0.5，0.5）为中心并且半径为0.5
            # 在轴坐标中。
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type 必须是'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon 给出以1为中心的半径为1的多边形
                # （0，0），但我们希望以（0.5，
                #   0.5）的坐标轴。
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta

#LCRFM模型作图
N = num_clusters
theta = radar_factory(N, frame='polygon')
data = r.to_numpy()
fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1,
                         subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

# 去掉最后一列
case_data = data[:, :-1]
# 设置纵坐标不可见
ax.get_yaxis().set_visible(False)
# 图片标题
title = "Radar Chart for Different Means"
ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
             horizontalalignment='center', verticalalignment='center')
for d in case_data:
    # 画边
    ax.plot(theta, d)
    # 填充颜色
    ax.fill(theta, d, alpha=0.05)
# 设置纵坐标名称
ax.set_varlabels(features)
# 添加图例
labels = ["CustomerCluster_" + str(i) for i in range(1,6)]
legend = ax.legend(labels, loc=(0.9, .75), labelspacing=0.1)

plt.show()

#RFM模型作图
theta = radar_factory(3, frame='polygon')
data = rr.to_numpy()
fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1,
                         subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

# 去掉最后一列
case_data = data[:, :-1]
# 设置纵坐标不可见
ax.get_yaxis().set_visible(False)
# 图片标题
title = "Radar Chart for Different Means"
ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
             horizontalalignment='center', verticalalignment='center')
for d in case_data:
    # 画边
    ax.plot(theta, d)
    # 填充颜色
    ax.fill(theta, d, alpha=0.05)
# 设置纵坐标名称
ax.set_varlabels(['R','F','M'])
# 添加图例
labels = ["CustomerCluster_" + str(i) for i in range(1,6)]
legend = ax.legend(labels, loc=(0.9, .75), labelspacing=0.1)

plt.show()

#DBSCAN模型对LCRFM特征进行计算
from sklearn.cluster import DBSCAN
# db = DBSCAN(eps=10,min_samples=2).fit(data_db)

# Kagging debug
db = DBSCAN(eps=10,min_samples=2).fit(data_db.sample(10000))

DBSCAN_labels = db.labels_

print("DBSCAN模型对每个样本预测的群体标签：：")
print(DBSCAN_labels)
