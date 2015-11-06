# coding: utf-8

import math
import numpy as np
import scipy
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt

from collections import defaultdict
from layout import ForceLayout


class TDA(object):

    def __init__(self, distance, filters, K):
        '''
        distance: 距离函数 f(s,t)  s,t 为data中的向量
        filters:  [(过滤函数, interval, overlap) ...] 
                  其中过滤函数为 f(d, i),   d: 原始数据， i: 要计算的数据点的索引
        K:        单链接聚类直方图间隔数
        '''

        self.distance = distance
        self.filters = filters
        self.K = K


    def _bin(self):
        '''
        使用filter函数进行数据点的分箱
        '''
        def cartprod(sets):
            '''
            笛卡尔积
            '''
            if len(sets) > 1:
                for r in cartprod(sets[1:]):
                    for elem in sets[0]:
                        yield (elem, ) + r
            else:
                for elem in sets[0]:
                    yield (elem, )

        
        fvalues = [] # filter 函数值
        fstep = []   # filter 值切割间隔
        for filterfunc, interval, overlap in self.filters:
            fvals = [ filterfunc(self.data, i) for i in range(len(self.data))]
            fvalues.append(fvals)
            fstep.append( (max(fvals) - min(fvals)) / interval / (1 - overlap))

        bins = defaultdict(list)
        for i in range(len(self.data)):
            coordranges = []
            for j, (filterfunc, interval, overlap) in enumerate(self.filters):
                uncovered = fstep[j] * (1 - overlap)
                fvalue = fvalues[j][i]
                binstart = math.floor((fvalue - fstep[j]) / uncovered) + 1
                binend   = math.floor(fvalue / uncovered)
                coordranges.append(range(binstart, binend + 1))

            for coord in cartprod(coordranges):
                bins[coord].append(i)

        return bins


    def _cluster(self, bin):
        '''
        箱内聚类，使用 single linkeage 
        bin: 序号数组
        return: 序号集合数组
        '''
        if len(bin) <= 3:
            return [set(bin)]

        if isinstance(self.distance, IndexDistance):
            def metric(a, b):
                i, j = int(a[0]), int(b[0])
                return self.distance.func(self.data, i, j)
        else:
            def metric(a, b):
                pa, pb = self.data[int(a[0])], self.data[int(b[0])]
                return  self.distance(pa, pb)

        y = [[i] for i in bin]
        z = hac.linkage(y, method='single', metric=metric)
        transitions = [i[2] for i in z]
        intv = (max(transitions) - min(transitions)) / self.K

        # 特殊情况，距离完全相等
        if intv == 0.0:
            return [set(bin)]

        thresh = 0
        indexstart = indexend = math.floor(transitions[0] / intv)
        for t in transitions[1:]:
            index = math.floor(t / intv)
            if index < indexstart -1 or index > indexend + 1 :
                break
            indexstart = min(indexstart, index)
            indexend = max(indexend, index)
            thresh = t

        fcluster = hac.fcluster(z, thresh, criterion='distance')
        clusters = defaultdict(set)
        for binindex, clusterindex in zip(bin, fcluster):
            clusters[clusterindex].add(binindex)

        return list(clusters.values())


    def fit(self, data, Sk=1, Smin=0.32, Wk=1, Wmin=0.45, mincluster=0):
        '''
        data: 二维数组，数据
        Sk, Smin: 表示对点权重的微调，Sk对全体调整，Smin对相对较弱的点进行调节
        Wk, Wmin: 表示对连接线强度的微调，Wk对全体连接线强度调整，Wmin对弱连接进行调整
        '''
        self.data = data
        bins = self._bin()

        # 计算箱子数
        binindexsets = [set() for _ in self.filters]
        for binindex in bins.keys():
            for i, bi in enumerate(binindex):
                binindexsets[i].add(bi)
        binums = [len(s) for s in binindexsets]  

        # 箱内聚类
        clusters = []
        for bin in bins.values():
            print(len(bin))
            clusters.extend(self._cluster(bin))
        # cluster数量过滤
        clusters = [c for c in clusters if len(c) > mincluster]
        linkages = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                w = len(clusters[i] & clusters[j])
                if w:
                    linkages.append((i, j, w))

        ## 力学布局
        def scaleto(d, a, b):
            min_, max_ = 1, max(d)
            k_ = (a - b) / (min_ - max_) if max_ > min_ else 0.0
            return [ (i-max_) * k_ + b for i in d]

        n = len(clusters)

        # 尺寸(归一化)
        S = scaleto([len(c) for c in clusters], Smin, 1)
        S = np.array(S).reshape(n, 1)

        # 连接阵
        W = np.zeros((n, n))
        wij_tr_arr = scaleto([wij for i, j, wij in linkages], Wmin, 1)
        for (i, j, wij), wij_tr in zip(linkages, wij_tr_arr):
            W[i, j] = W[j, i] = wij_tr

        layout = ForceLayout(Gk=0.006, Kq=5 * Sk, Lk=0.12 * Wk, Rc=0.08, M=0.3)
        positions = layout.run(n, S, W, dt=0.2, stopenergy=0.1, randomInit=True)

        self.binums = binums        # filter分箱子的个数，一维数组
        self.clusters = clusters    # 二级分类结果
        self.linkages = linkages
        self.positions = positions  # 最终点坐标


    def dye(self, color=None, title=None, C=1, Pk=1, Pmin=0.2, figsize=6, agg=np.mean):
        '''
        染色，画图
        color: 染色函数 f(d, i) , return float value
        title: 标题
        C     : 正实数, 控制整体颜色亮度的一个值，默认为1 表示原色，值越大整体亮度越高
        Pk    : 调节所有点的大小
        Pmin  : 调节所有较小点的大小
        figsize: 图片尺寸
        agg  : 颜色的聚合函数(每个点包含了很多歌颜色值), like max, min ... , default=np.mean
        '''
        def scaleto(d, a, b):
            min_, max_ = min(d), max(d)
            k_ = (a - b) / (min_ - max_) if max_ > min_ else 0.0
            return [ (i-max_) * k_ + b for i in d]

        # 坐标与画布尺寸
        X = [ x for x, y in self.positions]
        Y = [ y for x, y in self.positions]
        margin = max([abs(i) for i in X+Y]) * 1.2

        # 点大小
        clusternum = np.array([len(i) for i in self.clusters])
        area = np.array(scaleto(clusternum, Pmin, 1)) * (60 * Pk * figsize / margin) **2

        # 颜色
        if color:
            colors = [color(self.data, i) for i in range(len(self.data))]
        else:
            colors = [0 for i in range(len(self.data))]
        normcolors = np.array(scaleto(colors, 0, 1)) ** (1/C)
        clustercolors = [ agg([normcolors[i] for i in c]) for c in self.clusters]

        plt.figure(figsize=(figsize, figsize))
        plt.xlim(-margin, margin)
        plt.ylim(-margin, margin)

        plt.scatter(X, Y, s=area, c=clustercolors,  alpha=1, cmap=plt.cm.rainbow)
        if title:
            plt.title(title)


    def dyes(self, colors, titles=[], Pk=1, Pmin=0.2, figsize=6, width=3, agg=np.mean):
        '''
        染色，画图(多图)
        colors: 染色函数的数组 [f(d, i) , ... ] ,  f return float value
        titles: 子图的title
        Pk    : 调节所有点的大小
        Pmin  : 调节所有较小点的大小
        width : 每行的最大子图数量
        figsize: 图片尺寸
        agg  : 颜色的聚合函数(每个点包含了很多歌颜色值), like max, min ... , default=np.mean
        '''
        def scaleto(d, a, b):
            min_, max_ = min(d), max(d)
            k_ = (a - b) / (min_ - max_) if max_ > min_ else 0.0
            return [ (i-max_) * k_ + b for i in d]

        # 坐标与画布尺寸
        X = [ x for x, y in self.positions]
        Y = [ y for x, y in self.positions]
        margin = max([abs(i) for i in X+Y]) * 1.2

        # 点大小
        clusternum = np.array([len(i) for i in self.clusters])
        area = np.array(scaleto(clusternum, Pmin, 1)) * (60 * Pk * figsize / margin) **2

        height = math.ceil(len(colors) / width)
        fig, ax = plt.subplots(height, width, sharex=False, sharey=True)
        fig.set_figheight(figsize*height)
        fig.set_figwidth(figsize*width)
        axes = ax.ravel()

        for i, color in enumerate(colors):
            # 颜色
            if color:
                colorvalues = [color(self.data, i) for i in range(len(self.data))]
            else:
                colorvalues = [0 for i in range(len(self.data))]
            normcolors = scaleto(colorvalues, 0, 1)
            clustercolors = [ agg([normcolors[i] for i in c]) for c in self.clusters]

            axes[i].set_xlim(-margin, margin)
            axes[i].set_ylim(-margin, margin)
            axes[i].scatter(X, Y, s=area, c=clustercolors, alpha=1, cmap=plt.cm.rainbow)
            axes[i].set_title(titles[i] if i < len(titles) else '')


class IndexDistance(object):
    '''
    TDA 接受的默认距离函数为 f(v1, v2) 的形式 (其中v1,v2 是两个数据点)
    如果想要使用 index 的形式 f(d, i, j) 的形式 (其中d为总数据集，i j 分别为两个数据点的序号), 
    则可以用 IndexDistance 类来包装 f(d, i, j) 函数
    '''
    def __init__(self, func):
        '''
        func :  距离函数, f(d, i, j)
        '''
        self.func = func

    
