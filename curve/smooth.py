"""用逐段三次Hermite多项式平滑一串离散点。"""

import numpy as np


def curve_between(a, b, ta=None, tb=None, samples=20, endpoint=True):
    """生成从点a到点b的曲线；ta/tb是两端切向量，省略时得到直线。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape or samples < 2:
        raise ValueError("a、b必须是同维向量，samples至少为2")
    ta = b-a if ta is None else np.asarray(ta, float)
    tb = b-a if tb is None else np.asarray(tb, float)
    t = np.linspace(0., 1., samples, endpoint=endpoint)[:, None]
    return ((2*t**3-3*t**2+1)*a + (t**3-2*t**2+t)*ta +
            (-2*t**3+3*t**2)*b + (t**3-t**2)*tb)


def simplify_path(points, free):
    """只删除能被安全直线跨过的中间点，保留起点和终点。"""
    p, kept = np.asarray(points, float), [0]
    if p.ndim != 2 or len(p) < 2:
        raise ValueError("points必须是至少两个点组成的二维数组")
    while kept[-1] < len(p)-1:
        i, j = kept[-1], len(p)-1
        while j > i+1:
            n = max(2, int(4*np.linalg.norm(p[j]-p[i]))+1)
            if free(np.linspace(p[i], p[j], n)):
                break
            j -= 1
        kept.append(j)
    return p[kept]


def smooth_path(points, samples=20, tension=.6, free=None, retries=8):
    """连接相邻点；free(curve)可检查曲线是否碰撞，失败时自动收紧弯曲。"""
    p = np.asarray(points, float)
    if p.ndim != 2 or len(p) < 2:
        raise ValueError("points必须是至少两个点组成的二维数组")
    edge = p[1:]-p[:-1]
    length = np.linalg.norm(edge, axis=1)
    direction = edge/np.maximum(length[:, None], 1e-12)
    tangent = np.empty_like(p)
    tangent[0], tangent[-1] = edge[0], edge[-1]
    turn = direction[:-1]+direction[1:]
    turn /= np.maximum(np.linalg.norm(turn, axis=1, keepdims=True), 1e-12)
    tangent[1:-1] = turn*np.minimum(length[:-1], length[1:])[:, None]
    tangent *= tension

    # 相邻两段共享同一个节点切向量，因此节点处方向连续；靠墙时只缩短切向量。
    for _ in range(retries):
        bad = [i for i in range(len(p)-1)
               if free is not None and not free(curve_between(p[i], p[i+1], tangent[i], tangent[i+1], samples))]
        if not bad:
            break
        tangent[np.unique([j for i in bad for j in (i, i+1)])] *= .5
    parts = [curve_between(p[i], p[i+1], tangent[i], tangent[i+1], samples, False)
             for i in range(len(p)-1)]
    curve = np.vstack((*parts, p[-1]))
    if free is not None and not free(curve):
        raise ValueError("原折线路径离墙太近，无法安全曲线化")
    return curve
