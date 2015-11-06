import math
import numpy as np


class ForceLayout(object):
    '''
    力学场景布置：
    1. 原点(0, 0)位置处放置引力源，所有单位受到指向原点的引力 = Gk * S(i) * r
    2. 所有单位带同性电荷, 斥力 = Kq * S(i) * S(j) / r^2
    3. 部分单位间有弹簧连接，相互引力 = Lk * W(i,j) * r
    4. 存在空气阻力 = Rc * v^2，阻力系数为Rc， v为速度
    5. 质量(加速度质量) = M * S(i)

    环境变量设定：
        Gk, Kq, Lk, Rc, M 为常量
        S(i) 为单位i的大小(归一化)
        W(i, j) 为单位i与j之间的连接强度(归一化)

    '''

    def __init__(self, Gk, Kq, Lk, Rc, M):
        self.Gk = Gk
        self.Kq = Kq
        self.Lk = Lk
        self.Rc = Rc
        self.M  = M


    def run(self, *args, **argv):
        argv['debugStep'] = 0
        it = self.run_inner(*args, **argv)
        positions, energy, loop = it.send(None)
        return positions


    def run_inner(self, n, S, W, dt, stopenergy, randomInit=True, initRect=100, maxLoop=2400, debugStep=0):
        '''
        n       : 个数
        S       : 一维列向量，表示每个单位的权重
        W       : n * n 矩阵, wij 表示弹簧强度
        dt      : 每帧时长
        stopenergy : 停止条件，平均能量小于stopenergy
        randomInit : 初始化位置时是否随机
        initRect: 随机初始化位置的范围
        maxLoop : 最大运行次数
        debugStep: 调试状态，没过这么多帧输出一次结果， 0 表示关闭调试状态

        return  : (各单位坐标坐标, 平均能量, 运行步数) 的迭代器
        '''
        # Aij = Si * Sj
        A = np.dot(S, np.ones((n, 1)).T) * np.dot(np.ones((n, 1)), S.T)

        # 位移
        if randomInit:
            rx = (np.random.rand(n, 1) - 0.5) * initRect
            ry = (np.random.rand(n, 1) - 0.5) * initRect
        else:
            rx = initRect / 2 * np.cos( 2 * np.pi / n * np.arange(n).reshape(n, 1) )
            ry = initRect / 2 * np.sin( 2 * np.pi / n * np.arange(n).reshape(n, 1) )

        # 速度
        vx = np.zeros((n, 1))
        vy = np.zeros((n, 1))

        for loop in range(maxLoop):
            # 受力
            fx, fy = self.getforce(n, S, A, W, rx, ry, vx, vy)

            # 能量=势能+动能(量级)
            energy = (vx**2 + vy**2).mean() + (fx**2 + fy**2).mean()
            if energy < stopenergy:
                yield list(zip(rx.ravel(), ry.ravel())), energy, loop
                raise StopIteration()

            dt_ = min(dt , 10 / math.sqrt(energy))   # 防止跑飞

            rx += dt_ * vx + dt_ **2 /2 /self.M / S * fx
            ry += dt_ * vy + dt_ **2 /2 /self.M / S * fy
            vx += dt_ * fx
            vy += dt_ * fy

            if debugStep > 0 and loop % debugStep == 0:
                yield list(zip(rx.ravel(), ry.ravel())), energy, loop

        raise Exception('Force Layout run over %d steps' % maxLoop)


    def getforce(self, n, S, A, W, rx, ry, vx, vy):
        O = np.ones((n, 1))
        rxi_rxj = np.dot(rx, O.T) - np.dot(O, rx.T)
        ryi_ryj = np.dot(ry, O.T) - np.dot(O, ry.T)

        mdist3 = (rxi_rxj ** 2 + ryi_ryj ** 2) ** 1.5
        mdist3[mdist3 < 0.001] =  0.01 * (mdist3[mdist3 < 0.001] ** (1.0/3.0)) # 库仑力极限距离0.1，小于此距离则当做0.1计算
        mdist3[range(n), range(n)] = np.inf     # 去除自己对自己的斥力作用

        # 受力初始化
        fx = np.zeros((n, 1))
        fy = np.zeros((n, 1))

        # 恒星引力
        fx += -self.Gk * S * rx
        fy += -self.Gk * S * ry

        # 库仑力
        fx += self.Kq * (A * rxi_rxj / mdist3).sum(axis=1).reshape(n, 1)
        fy += self.Kq * (A * ryi_ryj / mdist3).sum(axis=1).reshape(n, 1)

        # 弹簧力
        fx += -self.Lk * (W * rxi_rxj).sum(axis=1).reshape(n, 1)
        fy += -self.Lk * (W * ryi_ryj).sum(axis=1).reshape(n, 1)

        # 空气阻力
        vlen = (vx ** 2 + vy ** 2) ** 0.5
        fx += -self.Rc * vlen * vx
        fy += -self.Rc * vlen * vy

        return fx, fy
        
