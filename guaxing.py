#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guaxing.py  —  卦形几何生成器 / Gua-Figure Geometry Builder
==========================================================

把一个三角形分割成 2^(n-1) 个 *等积* 三角形（内部彼此对等，但不必等边），
并按「上爻占一半面积、每往下一爻减半、最底两爻各占一个最小格」的规则
把格子分配给 n 个爻。

    n = 1  →  2^0 = 1 格   2 种组合   阴 / 阳
    n = 2  →  2^1 = 2 格   4 种组合   四象
    n = 3  →  2^2 = 4 格   8 种组合   八卦        （= 你的 GeoGebra 图）
    n = 4  →  2^3 = 8 格  16 种组合
    n = 5  →  2^4 = 16 格 32 种组合
    n = 6  →  2^5 = 32 格 64 种组合   六十四卦

分配规则（从 GeoGebra 图反推并已核对全部八卦）：
    八卦 4 格：上爻 = 左上 + 右上（各 1/4，合 1/2）
               中爻 = 中央倒三角（1/4）
               初爻 = 底部三角（1/4）
    验证：乾=全阳 → 4 格全填；坤=全阴 → 4 格全空；
          坎(中爻独阳) → 只有中央格；震(初爻独阳) → 只有底格；
          艮(上爻独阳) → 只有左上+右上；离/巽/兑 皆合。
    四象 2 格：上爻 = 左半，初爻 = 右半
          验证：少阳(上阳下阴) → 左半有花纹；少阴(上阴下阳) → 右半有花纹。

分割算法（保证每个小三角形面积严格相等）：
    median_split(T)   : 顶边中点 → 顶点，一分为二（面积各 1/2）
    midpoint_split(T) : 三边中点全连，一分为四（面积各 1/4）
                        → 左上 TL、右上 TR、中央 C（倒置）、底部 B
    subdiv(T, m)      : 把 T 分成 2^m 个等积三角形
                        m>=2 用 midpoint_split 再递归 m-2；m==1 用 median_split

用法
----
    python guaxing.py                 启动 GUI（需要 PyQt6）
    python guaxing.py --selftest      纯几何自检，不需要 PyQt6
    python guaxing.py --batch OUTDIR --n 6 --fmt svg
                                      命令行批量导出全部 2^n 张图

作者：为 Andrew 的 geometry 项目而写。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

# =============================================================================
# 第 1 部分：纯几何内核（不依赖 Qt，可单独 import 使用）
# =============================================================================

EPS = 1e-9
SQRT3_2 = math.sqrt(3.0) / 2.0


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


@dataclass(frozen=True)
class Tri:
    """三角形。约定 p0-p1 为「底边」（与顶点 p2 相对的那条边）。

    median_split 从 p0-p1 的中点连到 p2；
    midpoint_split 以 p0-p1 为「上边」展开成 TL/TR/C/B。
    """
    p0: tuple
    p1: tuple
    p2: tuple

    @property
    def pts(self):
        return (self.p0, self.p1, self.p2)

    @property
    def area(self):
        (x0, y0), (x1, y1), (x2, y2) = self.p0, self.p1, self.p2
        return abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * 0.5

    @property
    def centroid(self):
        return ((self.p0[0] + self.p1[0] + self.p2[0]) / 3.0,
                (self.p0[1] + self.p1[1] + self.p2[1]) / 3.0)

    def contains(self, p, tol=1e-9):
        """重心坐标判定点是否在三角形内（含边）。"""
        (x0, y0), (x1, y1), (x2, y2) = self.p0, self.p1, self.p2
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < EPS:
            return False
        a = ((y1 - y2) * (p[0] - x2) + (x2 - x1) * (p[1] - y2)) / d
        b = ((y2 - y0) * (p[0] - x2) + (x0 - x2) * (p[1] - y2)) / d
        c = 1.0 - a - b
        return a >= -tol and b >= -tol and c >= -tol


def base_triangle(point_down=True, side=1.0):
    """等边三角形。数学坐标系（y 向上）。point_down = 尖朝下（同你的 GeoGebra 图）。"""
    h = side * SQRT3_2
    if point_down:
        # 上边在 y=0，顶点在下方
        return Tri((-side / 2, 0.0), (side / 2, 0.0), (0.0, -h))
    else:
        return Tri((-side / 2, 0.0), (side / 2, 0.0), (0.0, h))


def median_split(t: Tri, mirror=False):
    """中线二等分：面积各 1/2。返回 (第一半, 第二半)。"""
    m = _mid(t.p0, t.p1)
    a = Tri(t.p0, m, t.p2)
    b = Tri(m, t.p1, t.p2)
    return (b, a) if mirror else (a, b)


def oblique_split(t: Tri, mirror=False):
    """斜切二等分：底角 p0 → 对边 (p1,p2) 的中点。面积同样各 1/2。

    与 median_split 的差别只在「左右镜像不再是这条切线的对称操作」，
    因此最末两爻的两个最小格不会被整图镜像互换（见 symmetry_report）。
    """
    if mirror:
        m = _mid(t.p0, t.p2)
        return (Tri(t.p1, t.p0, m), Tri(t.p1, m, t.p2))
    m = _mid(t.p1, t.p2)
    return (Tri(t.p0, t.p1, m), Tri(t.p0, m, t.p2))


def split_two(t: Tri, mode="median", mirror=False):
    return (oblique_split(t, mirror) if mode == "oblique"
            else median_split(t, mirror))


def midpoint_split(t: Tri):
    """三边中点四等分：面积各 1/4。返回 (TL, TR, C, B)。

    TL/TR/B 与母三角形同向，C 为倒置的中央三角形。
    """
    m01 = _mid(t.p0, t.p1)   # 上边中点
    m02 = _mid(t.p0, t.p2)   # 左边中点
    m12 = _mid(t.p1, t.p2)   # 右边中点
    TL = Tri(t.p0, m01, m02)
    TR = Tri(m01, t.p1, m12)
    C = Tri(m02, m12, m01)   # 倒置：底边 m02-m12，顶点 m01
    B = Tri(m02, m12, t.p2)
    return TL, TR, C, B


def subdiv(t: Tri, m: int, mirror=False):
    """把三角形 t 分成 2^m 个面积严格相等的三角形。"""
    if m <= 0:
        return [t]
    if m == 1:
        return list(median_split(t, mirror))
    TL, TR, C, B = midpoint_split(t)
    if mirror:
        TL, TR = TR, TL
    out = []
    for child in (TL, TR, C, B):
        out.extend(subdiv(child, m - 2, mirror))
    return out


# ---------------------------------------------------------------- 爻位分配 --

def _assign(t: Tri, lines, out: dict, mirror: bool, last_cut="median"):
    """把三角形 t 按「首爻占一半、次爻占四分之一、其余递归」分给 lines。

    lines[0] 拿最大的区域，lines[-1] 与 lines[-2] 拿同样大的最小格。
    last_cut 只作用于最末两爻那一刀。
    """
    k = len(lines)
    if k == 1:
        out[lines[0]] = [t]
        return
    if k == 2:
        a, b = split_two(t, last_cut, mirror)
        out[lines[0]] = [a]
        out[lines[1]] = [b]
        return
    TL, TR, C, B = midpoint_split(t)
    if mirror:
        TL, TR = TR, TL
    m = k - 3                      # TL / TR / C 各含 2^m 个最小格
    out[lines[0]] = subdiv(TL, m, mirror) + subdiv(TR, m, mirror)
    out[lines[1]] = subdiv(C, m, mirror)
    _assign(B, lines[2:], out, mirror, last_cut)


def expected_group_sizes(n: int):
    """按爻分区时，从「大块」到「小块」依次应有多少个最小格。"""
    sizes, k = [], n
    while k > 2:
        sizes.append(2 ** (k - 2))
        sizes.append(2 ** (k - 3))
        k -= 2
    if k == 2:
        sizes.extend([1, 1])
    elif k == 1:
        sizes.append(1)
    return sizes


# 本系统只有这四层爻数。四爻、五爻不存在 ——
# 对半细分这条生法产出的形态数是 2 → 3 → 6，四和五不在名单上；
# 三爻之后不是加第四爻第五爻，而是把三爻整个叠一次变六爻。
GUA_LEVELS = (1, 2, 3, 6)
GUA_LEVEL_NAMES = {1: "一爻（阴阳）", 2: "二爻（四象）",
                   3: "三爻（八卦）", 6: "六爻（六十四卦）"}

YAO_NAMES = {
    1: ["爻"],
    2: ["初爻", "上爻"],
    3: ["初爻", "中爻", "上爻"],
    4: ["初爻", "二爻", "三爻", "上爻"],
    5: ["初爻", "二爻", "三爻", "四爻", "上爻"],
    6: ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"],
}


@dataclass
class Layout:
    n: int
    mode: str                 # "gua" 按爻分区 / "even" 纯等分
    cells: list               # list[Tri]，全局编号顺序
    cell_line: list           # 格号 → 爻号（0 = 初爻）；纯等分模式下为 -1
    lines: list               # 爻号 → 该爻的格号列表
    mirror: bool
    high_first: bool
    point_down: bool
    last_cut: str = "median"

    @property
    def n_cells(self):
        return len(self.cells)

    def yao_label(self, li):
        if li < 0:
            return "—"
        return YAO_NAMES[self.n][li]

    def cell_at(self, p):
        for i, t in enumerate(self.cells):
            if t.contains(p):
                return i
        return -1


def build_layout(n: int, mode="gua", mirror=False, high_first=True,
                 point_down=True, side=1.0, last_cut="median") -> Layout:
    """构造分割。n∈[1,6]（算法本身不限，只是名称表到 6）。"""
    if n < 1:
        raise ValueError("n 必须 >= 1")
    t0 = base_triangle(point_down, side)

    if mode == "even":
        cells = subdiv(t0, n - 1, mirror)
        return Layout(n, "even", cells, [-1] * len(cells),
                      [[] for _ in range(n)], mirror, high_first, point_down,
                      last_cut)

    order = list(range(n - 1, -1, -1)) if high_first else list(range(n))
    groups: dict = {}
    _assign(t0, order, groups, mirror, last_cut)

    cells, cell_line = [], []
    lines = [[] for _ in range(n)]
    for li in order:                       # 大块在前，编号稳定
        for t in groups[li]:
            lines[li].append(len(cells))
            cell_line.append(li)
            cells.append(t)
    return Layout(n, "gua", cells, cell_line, lines,
                  mirror, high_first, point_down, last_cut)


def validate_layout(lay: Layout):
    """几何自检。返回 (ok, [信息...])。"""
    msgs, ok = [], True
    n = lay.n
    want_cells = 2 ** (n - 1)
    if lay.n_cells != want_cells:
        ok = False
        msgs.append(f"✗ 格数 {lay.n_cells}，应为 2^({n}-1) = {want_cells}")
    else:
        msgs.append(f"✓ 格数 = 2^({n}-1) = {want_cells}")

    total = base_triangle(lay.point_down).area
    areas = [c.area for c in lay.cells]
    s = sum(areas)
    if abs(s - total) > 1e-9 * max(1.0, total):
        ok = False
        msgs.append(f"✗ 面积总和 {s:.12f} ≠ 母三角形 {total:.12f}（有重叠或空隙）")
    else:
        msgs.append(f"✓ 面积总和 = 母三角形面积（无重叠、无空隙）")

    if areas:
        amin, amax = min(areas), max(areas)
        if amax - amin > 1e-9 * max(1.0, amax):
            ok = False
            msgs.append(f"✗ 各格面积不等：min={amin:.12g} max={amax:.12g}")
        else:
            msgs.append(f"✓ 各格面积严格相等 = {amax:.12g}"
                        f"（= 母面积 / {want_cells}）")

    # 质心互异 → 没有重复格
    cents = {(round(c.centroid[0], 12), round(c.centroid[1], 12))
             for c in lay.cells}
    if len(cents) != lay.n_cells:
        ok = False
        msgs.append("✗ 有重复的格子")
    else:
        msgs.append("✓ 无重复格子")

    if lay.mode == "gua":
        order = list(range(n - 1, -1, -1)) if lay.high_first else list(range(n))
        got = [len(lay.lines[li]) for li in order]
        want = expected_group_sizes(n)
        if got != want:
            ok = False
            msgs.append(f"✗ 爻区格数 {got}，应为 {want}")
        else:
            desc = " + ".join(str(x) for x in want)
            msgs.append(f"✓ 爻区格数（大→小）= {desc} = {want_cells}")
        for li in range(n):
            frac = len(lay.lines[li]) / want_cells
            msgs.append(f"    {lay.yao_label(li)}：{len(lay.lines[li])} 格"
                        f"，占 {frac:.6g} 面积")

        r = symmetry_report(lay)
        msgs.append("")
        msgs.append("—— 图形唯一性 ——")
        if r["distinct_fixed"]:
            msgs.append(f"✓ 固定摆放时，{r['total']} 个图形两两不同"
                        f"（各爻占互不相交的非空格组，映射是单射）")
        else:
            ok = False
            msgs.append(f"✗ 有图形重复")
        msgs.append(f"  没有第 {r['total']+1} 种：按爻分区的填色恰好 2^{n} "
                    f"= {r['total']} 种，一个不多一个不少")
        msgs.append(f"  保持这套分割的对称操作：{'、'.join(r['syms'])}")
        if r["classes"] == r["total"]:
            msgs.append(f"✓ 连「可翻转旋转后重合」也算，仍是 {r['total']}/"
                        f"{r['total']} 个互不全等")
        else:
            msgs.append(f"! 在全等（允许翻转旋转）意义下只有 {r['classes']}/"
                        f"{r['total']} 类：以下 {len(r['pairs'])} 组互为镜像")
            for grp in r["pairs"][:8]:
                msgs.append("      " + " ↔ ".join(
                    f"{gua_name(c, n)}({code_bin(c, n)})" for c in grp))
            if len(r["pairs"]) > 8:
                msgs.append(f"      …… 共 {len(r['pairs'])} 组")
            msgs.append("      成因：末两爻是同一刀切出的左右两半，整图竖轴镜像"
                        "正好把它们互换")
            msgs.append("      解法：把「末两爻切法」改为斜切 → 全部 "
                        f"{r['total']} 个互不全等")
        msgs.append(f"  若不受爻分区约束（均分模式），同一套 {want_cells} 格"
                    f"可承载 2^{want_cells} = {r['free_fillings']} 种填色")
    return ok, msgs


def _tri_key(t: Tri, nd=9):
    return tuple(sorted((round(p[0], nd), round(p[1], nd)) for p in t.pts))


def _d3_ops(tri: Tri):
    """外框等边三角形的 6 个自身对称（3 旋转 + 3 镜像）。"""
    cx = sum(p[0] for p in tri.pts) / 3.0
    cy = sum(p[1] for p in tri.pts) / 3.0
    ops = {}
    for ang in (0, 120, 240):
        a = math.radians(ang)
        ops[f"旋转{ang}°"] = (lambda p, a=a: (
            cx + (p[0] - cx) * math.cos(a) - (p[1] - cy) * math.sin(a),
            cy + (p[0] - cx) * math.sin(a) + (p[1] - cy) * math.cos(a)))
    for th, nm in ((90, "竖轴镜像"), (30, "斜轴镜像30°"), (150, "斜轴镜像150°")):
        a = math.radians(2 * th)
        ops[nm] = (lambda p, a=a: (
            cx + (p[0] - cx) * math.cos(a) + (p[1] - cy) * math.sin(a),
            cy + (p[0] - cx) * math.sin(a) - (p[1] - cy) * math.cos(a)))
    return ops


def symmetry_report(lay: Layout):
    """回答两个问题：2^n 个图形是否两两不同？在全等（可翻转旋转）意义下呢？

    返回 dict：
      distinct_fixed  — 固定摆放时是否两两不同（恒为 True，见下方证明性检查）
      syms            — 把这套分割映回自身的对称操作名
      classes         — 全等意义下的等价类数
      pairs           — 互为全等的卦码配对
      free_fillings   — 若不受爻分区约束，同一套格子能承载多少种填色
    """
    n = lay.n
    keys = [_tri_key(t) for t in lay.cells]
    kset, idx = set(keys), {k: i for i, k in enumerate(keys)}

    syms, perms = [], []
    for name, f in _d3_ops(base_triangle(lay.point_down)).items():
        mk = [tuple(sorted((round(f(p)[0], 9), round(f(p)[1], 9))
                           for p in t.pts)) for t in lay.cells]
        if set(mk) == kset:
            syms.append(name)
            perms.append([idx[k] for k in mk])

    fills = {}
    for code in range(2 ** n):
        b = bits_of(code, n)
        fills[code] = frozenset(ci for li in range(n) if b[li]
                                for ci in lay.lines[li])
    distinct_fixed = len(set(fills.values())) == 2 ** n

    canon, classes = {}, {}
    for code in range(2 ** n):
        key = min(tuple(sorted(frozenset(p[x] for x in fills[code])))
                  for p in perms)
        canon[code] = key
        classes.setdefault(key, []).append(code)
    pairs = [tuple(v) for v in classes.values() if len(v) > 1]

    return {"distinct_fixed": distinct_fixed, "syms": syms,
            "classes": len(classes), "total": 2 ** n, "pairs": pairs,
            "free_fillings": 2 ** lay.n_cells}


# =============================================================================
# 第 2 部分：卦名 / 编码
# =============================================================================

# 爻码约定：整数 code，bit i = 第 i 爻（0 = 初爻），1 = 阳，0 = 阴。

TRIGRAM = {7: "乾", 3: "兑", 5: "离", 1: "震",
           6: "巽", 2: "坎", 4: "艮", 0: "坤"}
TRIGRAM_SYMBOL = {7: "☰", 3: "☱", 5: "☲", 1: "☳",
                  6: "☴", 2: "☵", 4: "☶", 0: "☷"}
# Andrew 的连山链序：艮山→巽风→坎水→坤地/海→兑泽→震雷→离火→乾天→艮
LIANSHAN_CHAIN = [4, 6, 2, 0, 3, 1, 5, 7]
XIANTIAN_ORDER = [7, 3, 5, 1, 6, 2, 4, 0]      # 乾兑离震巽坎艮坤

NAMES_1 = {1: "阳", 0: "阴"}
NAMES_2 = {0b11: "太阳", 0b01: "少阴", 0b10: "少阳", 0b00: "太阴"}

# 六十四卦：_HEX64[上卦][下卦]
_HEX64 = {
    "乾": {"乾": "乾为天", "兑": "天泽履", "离": "天火同人", "震": "天雷无妄",
           "巽": "天风姤", "坎": "天水讼", "艮": "天山遯", "坤": "天地否"},
    "兑": {"乾": "泽天夬", "兑": "兑为泽", "离": "泽火革", "震": "泽雷随",
           "巽": "泽风大过", "坎": "泽水困", "艮": "泽山咸", "坤": "泽地萃"},
    "离": {"乾": "火天大有", "兑": "火泽睽", "离": "离为火", "震": "火雷噬嗑",
           "巽": "火风鼎", "坎": "火水未济", "艮": "火山旅", "坤": "火地晋"},
    "震": {"乾": "雷天大壮", "兑": "雷泽归妹", "离": "雷火丰", "震": "震为雷",
           "巽": "雷风恒", "坎": "雷水解", "艮": "雷山小过", "坤": "雷地豫"},
    "巽": {"乾": "风天小畜", "兑": "风泽中孚", "离": "风火家人", "震": "风雷益",
           "巽": "巽为风", "坎": "风水涣", "艮": "风山渐", "坤": "风地观"},
    "坎": {"乾": "水天需", "兑": "水泽节", "离": "水火既济", "震": "水雷屯",
           "巽": "水风井", "坎": "坎为水", "艮": "水山蹇", "坤": "水地比"},
    "艮": {"乾": "山天大畜", "兑": "山泽损", "离": "山火贲", "震": "山雷颐",
           "巽": "山风蛊", "坎": "山水蒙", "艮": "艮为山", "坤": "山地剥"},
    "坤": {"乾": "地天泰", "兑": "地泽临", "离": "地火明夷", "震": "地雷复",
           "巽": "地风升", "坎": "地水师", "艮": "地山谦", "坤": "坤为地"},
}


def bits_of(code: int, n: int):
    """返回 [初爻, 二爻, ...] 的 0/1 列表。"""
    return [(code >> i) & 1 for i in range(n)]


def code_string(code: int, n: int):
    """自上爻到初爻的 ▅/▂ 串，方便肉眼比对。"""
    b = bits_of(code, n)
    return "".join("▅" if x else "▂" for x in reversed(b))


def gua_name(code: int, n: int):
    if n == 1:
        return NAMES_1[code & 1]
    if n == 2:
        return NAMES_2[code & 3]
    if n == 3:
        return TRIGRAM[code & 7]
    if n == 6:
        lower = TRIGRAM[code & 7]
        upper = TRIGRAM[(code >> 3) & 7]
        return _HEX64[upper][lower]
    return code_string(code, n)          # n = 4、5 无传统名，用爻串


def gua_symbol(code: int, n: int):
    if n == 3:
        return TRIGRAM_SYMBOL[code & 7]
    if n == 6:
        return TRIGRAM_SYMBOL[(code >> 3) & 7] + TRIGRAM_SYMBOL[code & 7]
    return ""


def order_codes(n: int, scheme="binary"):
    """批量导出的排列顺序。"""
    if n == 3 and scheme == "lianshan":
        return list(LIANSHAN_CHAIN)
    if n == 3 and scheme == "xiantian":
        return list(XIANTIAN_ORDER)
    if n == 6 and scheme == "lianshan":
        # 8×8：行 = 下卦按连山链序，列 = 上卦按连山链序
        return [lo + 8 * up for lo in LIANSHAN_CHAIN for up in LIANSHAN_CHAIN]
    if n == 6 and scheme == "xiantian":
        return [lo + 8 * up for lo in XIANTIAN_ORDER for up in XIANTIAN_ORDER]
    return list(range(2 ** n))


# =============================================================================
# 第 3 部分：绘图模型（与 Qt 无关的绘图指令，SVG / PNG / 屏幕共用同一套）
# =============================================================================

def rot_pt(p, deg):
    """绕原点旋转（数学坐标系，y 向上；正角 = 逆时针）。"""
    if deg % 360 == 0:
        return p
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def rotate_ops(ops, deg):
    """只转位置，不转文字本身 —— 图形跟着转，字仍然正着看。"""
    if deg % 360 == 0:
        return ops
    out = []
    for op in ops:
        k = op[0]
        if k == "poly":
            out.append((k, tuple(rot_pt(q, deg) for q in op[1]),
                        op[2], op[3], op[4]))
        elif k == "line":
            out.append((k, rot_pt(op[1], deg), rot_pt(op[2], deg), op[3], op[4]))
        elif k == "dot":
            out.append((k, rot_pt(op[1], deg), op[2], op[3]))
        elif k == "text":
            out.append((k, rot_pt(op[1], deg), op[2], op[3], op[4], op[5]))
        else:
            out.append(op)
    return out


DEFAULT_STYLE = {
    "yang_fill": "#cfe2f3",
    "yin_fill": "#dceaf7",
    "hatch_color": "#1a73c8",
    "line_color": "#1a73c8",
    "outer_color": "#1560ab",
    "vertex_outer": "#1560ab",
    "vertex_inner": "#6b6b6b",
    "text_color": "#222222",
    "bg": "#ffffff",
    "hatch_spacing": 0.055,
    "hatch_width": 0.010,
    "line_width": 0.009,
    "outer_width": 0.018,
    "vertex_r": 0.020,
    "font_size": 0.055,
    "title_size": 0.095,
    "auto_detail": True,      # 格子变小时，自动按 √面积 缩小交叉线间距/字号/顶点
}


@dataclass
class CellStyle:
    yang: bool = False
    name: str = ""
    fill: str = ""        # 空 = 用默认阴/阳色
    hatch: bool | None = None   # None = 跟随 yang


@dataclass
class Model:
    n: int = 3
    mode: str = "gua"
    code: int = 0
    mirror: bool = False
    high_first: bool = True
    point_down: bool = True
    last_cut: str = "median"     # "median" 中线切 / "oblique" 斜切（破镜像对称）
    rotation: int = 0            # 0 / 90 / 180 / 270，仅影响显示与导出
    title: str = ""
    style: dict = field(default_factory=lambda: dict(DEFAULT_STYLE))
    cells: list = field(default_factory=list)      # list[CellStyle]
    show_vertices: bool = True
    show_grid: bool = True
    show_index: bool = False
    show_names: bool = True
    show_title: bool = True
    layout: Layout = None

    # ------------------------------------------------------------------
    def rebuild(self, keep_styles=True):
        old = self.cells if keep_styles else []
        self.layout = build_layout(self.n, self.mode, self.mirror,
                                   self.high_first, self.point_down,
                                   last_cut=self.last_cut)
        n_cells = self.layout.n_cells
        new = [CellStyle() for _ in range(n_cells)]
        for i in range(min(len(old), n_cells)):
            new[i] = old[i]
        self.cells = new
        if self.mode == "gua":
            self.apply_code(self.code)
        return self

    def apply_code(self, code: int):
        """按卦码给各爻的格子上阴阳。"""
        self.code = code & ((1 << self.n) - 1)
        if self.mode != "gua":
            return
        b = bits_of(self.code, self.n)
        for li in range(self.n):
            for ci in self.layout.lines[li]:
                self.cells[ci].yang = bool(b[li])
        if not self.title or self.title in _autotitles(self.n):
            self.title = gua_name(self.code, self.n)

    def cell_fill(self, i):
        cs = self.cells[i]
        if cs.fill:
            return cs.fill
        return self.style["yang_fill"] if cs.yang else self.style["yin_fill"]

    def cell_hatch(self, i):
        cs = self.cells[i]
        return cs.yang if cs.hatch is None else cs.hatch

    # ------------------------------------------------------------------
    def detail_scale(self):
        """格子越小，细节（交叉线间距、字号、顶点）按 √面积 等比缩小。"""
        if not self.style.get("auto_detail", True):
            return 1.0
        return math.sqrt(4.0 / max(1, self.layout.n_cells))

    def bbox(self):
        xs, ys = [], []
        for t in self.layout.cells:
            for p in t.pts:
                q = rot_pt(p, self.rotation)
                xs.append(q[0]); ys.append(q[1])
        pad = 0.06
        top = max(ys) + pad
        if self.show_title and self.title:
            top += self.style["title_size"] * 1.6
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, top)

    def to_dict(self):
        return {
            "format": "guaxing/1",
            "n": self.n, "mode": self.mode, "code": self.code,
            "mirror": self.mirror, "high_first": self.high_first,
            "point_down": self.point_down, "last_cut": self.last_cut,
            "rotation": self.rotation, "title": self.title,
            "style": self.style,
            "show": {"vertices": self.show_vertices, "grid": self.show_grid,
                     "index": self.show_index, "names": self.show_names,
                     "title": self.show_title},
            "cells": [{"yang": c.yang, "name": c.name,
                       "fill": c.fill, "hatch": c.hatch} for c in self.cells],
        }

    @staticmethod
    def from_dict(d):
        m = Model()
        m.n = int(d.get("n", 3))
        m.mode = d.get("mode", "gua")
        m.mirror = bool(d.get("mirror", False))
        m.high_first = bool(d.get("high_first", True))
        m.point_down = bool(d.get("point_down", True))
        m.last_cut = d.get("last_cut", "median")
        m.rotation = int(d.get("rotation", 0)) % 360
        m.style = dict(DEFAULT_STYLE); m.style.update(d.get("style", {}))
        sh = d.get("show", {})
        m.show_vertices = sh.get("vertices", True)
        m.show_grid = sh.get("grid", True)
        m.show_index = sh.get("index", False)
        m.show_names = sh.get("names", True)
        m.show_title = sh.get("title", True)
        m.code = int(d.get("code", 0))
        m.rebuild(keep_styles=False)
        for i, c in enumerate(d.get("cells", [])):
            if i < len(m.cells):
                m.cells[i] = CellStyle(bool(c.get("yang", False)),
                                       c.get("name", ""), c.get("fill", ""),
                                       c.get("hatch", None))
        m.title = d.get("title", m.title)
        return m


def _autotitles(n):
    return {gua_name(c, n) for c in range(2 ** n)}


# ----------------------------------------------------------------- 交叉线 --

def hatch_segments(shape, spacing: float, angles=(45.0, -45.0)):
    """把一组等距斜线精确裁剪到多边形内，返回 [(p, q), ...]（矢量，可直接进 SVG）。

    shape 可以是 Tri，也可以是任意（凸）多边形的点串 —— 圆的扇形、
    方的格子都走同一条路，所以三角形 / 圆 / 方的填色风格完全一致。
    """
    pts = list(shape.pts) if isinstance(shape, Tri) else list(shape)
    n = len(pts)
    segs = []
    for deg in angles:
        th = math.radians(deg)
        nx, ny = -math.sin(th), math.cos(th)      # 法向
        dx, dy = math.cos(th), math.sin(th)       # 方向
        proj = [p[0] * nx + p[1] * ny for p in pts]
        lo, hi = min(proj), max(proj)
        k0 = math.ceil(lo / spacing)
        k1 = math.floor(hi / spacing)
        for k in range(k0, k1 + 1):
            c = k * spacing
            # 半开区间穿越规则：顶点正好落在线上时只算一次，
            # 交点数因此必为偶数，凹形（太极月牙、四旋弯臂）也不会配错对。
            hits = []
            for i in range(n):
                a, b = pts[i], pts[(i + 1) % n]
                da = a[0] * nx + a[1] * ny - c
                db = b[0] * nx + b[1] * ny - c
                if (da < 0) == (db < 0):
                    continue
                if abs(da - db) < EPS:
                    continue
                u = da / (da - db)
                hits.append((a[0] + u * (b[0] - a[0]),
                             a[1] + u * (b[1] - a[1])))
            if len(hits) < 2 or len(hits) % 2:
                continue          # 奇数 = 数值退化，跳过这一条，绝不画错
            ts = sorted((h[0] * dx + h[1] * dy, h) for h in hits)
            for j in range(0, len(ts) - 1, 2):
                p, q = ts[j][1], ts[j + 1][1]
                if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 1e-14:
                    segs.append((p, q))
    return segs


# --------------------------------------------------------------- 绘图指令 --

def build_ops(m: Model):
    """生成与输出介质无关的绘图指令列表（模型坐标，y 向上）。"""
    st = m.style
    ops = []
    lay = m.layout
    ds = m.detail_scale()                       # 细节缩放
    dsw = max(ds, 0.55)                         # 线宽不要缩得看不见
    hatch_gap = max(0.012, st["hatch_spacing"] * ds)
    vr = st["vertex_r"] * max(ds, 0.42)
    fs = st["font_size"] * max(ds, 0.35)

    # 1) 填色
    for i, t in enumerate(lay.cells):
        ops.append(("poly", t.pts, m.cell_fill(i), None, 0.0))

    # 2) 交叉线（阳）
    for i, t in enumerate(lay.cells):
        if m.cell_hatch(i):
            for p, q in hatch_segments(t, hatch_gap):
                ops.append(("line", p, q, st["hatch_color"],
                            st["hatch_width"] * dsw))

    # 3) 内部格线
    if m.show_grid:
        seen = set()
        for t in lay.cells:
            pts = t.pts
            for i in range(3):
                a, b = pts[i], pts[(i + 1) % 3]
                key = tuple(sorted([(round(a[0], 10), round(a[1], 10)),
                                    (round(b[0], 10), round(b[1], 10))]))
                if key in seen:
                    continue
                seen.add(key)
                ops.append(("line", a, b, st["line_color"],
                            st["line_width"] * dsw))

    # 4) 外框
    outer = base_triangle(m.point_down)
    ops.append(("poly", outer.pts, None, st["outer_color"], st["outer_width"]))

    # 5) 顶点圆点
    if m.show_vertices:
        outer_pts = {(round(p[0], 10), round(p[1], 10)) for p in outer.pts}
        allpts = {}
        for t in lay.cells:
            for p in t.pts:
                allpts[(round(p[0], 10), round(p[1], 10))] = p
        for key, p in allpts.items():
            col = st["vertex_outer"] if key in outer_pts else st["vertex_inner"]
            ops.append(("dot", p, vr, col))

    # 6) 格内文字
    for i, t in enumerate(lay.cells):
        txt = []
        if m.show_index:
            txt.append(str(i))
        if m.show_names and m.cells[i].name:
            txt.append(m.cells[i].name)
        if txt:
            c = t.centroid
            ops.append(("text", c, " ".join(txt), fs,
                        st["text_color"], "middle"))

    ops = rotate_ops(ops, m.rotation)

    # 7) 标题（画在图形上方的留白里，不压住顶边）
    if m.show_title and m.title:
        x0, y0, x1, y1 = m.bbox()
        ops.append(("text", ((x0 + x1) / 2, y1 - st["title_size"] * 0.85),
                    m.title, st["title_size"], st["text_color"], "middle"))
    return ops


# ------------------------------------------------------------------- SVG --

CJK_FONT = ("'Microsoft YaHei','Noto Sans CJK SC','PingFang SC',"
            "'Source Han Sans SC',sans-serif")


def ops_to_svg(ops, bbox, px=900, bg="#ffffff", title="guaxing"):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    s = px / max(w, h)
    W, H = w * s, h * s

    def X(x):
        return (x - x0) * s

    def Y(y):
        return (y1 - y) * s      # 翻转 y

    out = [f'<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}" '
           f'height="{H:.2f}" viewBox="0 0 {W:.2f} {H:.2f}">',
           f'<title>{_esc(title)}</title>',
           f'<rect width="{W:.2f}" height="{H:.2f}" fill="{bg}"/>']
    for op in ops:
        k = op[0]
        if k == "poly":
            _, pts, fill, stroke, lw = op
            d = " ".join(f"{X(p[0]):.3f},{Y(p[1]):.3f}" for p in pts)
            f = fill if fill else "none"
            st_ = (f' stroke="{stroke}" stroke-width="{lw*s:.3f}" '
                   f'stroke-linejoin="round"') if stroke else ' stroke="none"'
            out.append(f'<polygon points="{d}" fill="{f}"{st_}/>')
        elif k == "line":
            _, p, q, col, lw = op
            out.append(f'<line x1="{X(p[0]):.3f}" y1="{Y(p[1]):.3f}" '
                       f'x2="{X(q[0]):.3f}" y2="{Y(q[1]):.3f}" '
                       f'stroke="{col}" stroke-width="{lw*s:.3f}" '
                       f'stroke-linecap="round"/>')
        elif k == "dot":
            _, p, r, col = op
            out.append(f'<circle cx="{X(p[0]):.3f}" cy="{Y(p[1]):.3f}" '
                       f'r="{r*s:.3f}" fill="{col}" stroke="#ffffff" '
                       f'stroke-width="{0.004*s:.3f}"/>')
        elif k == "text":
            _, p, txt, size, col, anchor = op
            out.append(f'<text x="{X(p[0]):.3f}" y="{Y(p[1]):.3f}" '
                       f'font-size="{size*s:.3f}" fill="{col}" '
                       f'font-family="{CJK_FONT}" text-anchor="{anchor}" '
                       f'dominant-baseline="central">{_esc(txt)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def model_to_svg(m: Model, px=900):
    return ops_to_svg(build_ops(m), m.bbox(), px, m.style["bg"],
                      m.title or "guaxing")


def sheet_to_svg(models, cols, px_each=320, gap=0.10, labels=None):
    """把多个 Model 排成一张总图（纯 SVG 拼版，坐标平移即可）。"""
    if not models:
        return ""
    bbs = [m.bbox() for m in models]
    w = max(b[2] - b[0] for b in bbs)
    h = max(b[3] - b[1] for b in bbs)
    rows = (len(models) + cols - 1) // cols
    cw, ch = w * (1 + gap), h * (1 + gap)
    total = (0.0, 0.0, cw * cols, ch * rows)
    s = px_each * cols / (cw * cols)

    all_ops = []
    for idx, m in enumerate(models):
        r, c = divmod(idx, cols)
        bx0, by0, bx1, by1 = m.bbox()
        ox = c * cw + (cw - (bx1 - bx0)) / 2 - bx0
        oy = (rows - 1 - r) * ch + (ch - (by1 - by0)) / 2 - by0
        for op in build_ops(m):
            all_ops.append(_shift_op(op, ox, oy))
    return ops_to_svg(all_ops, total, px_each * cols,
                      models[0].style["bg"], "guaxing sheet")


def _shift_op(op, dx, dy):
    k = op[0]
    if k == "poly":
        return (k, tuple((p[0] + dx, p[1] + dy) for p in op[1]), op[2], op[3], op[4])
    if k == "line":
        return (k, (op[1][0] + dx, op[1][1] + dy),
                (op[2][0] + dx, op[2][1] + dy), op[3], op[4])
    if k == "dot":
        return (k, (op[1][0] + dx, op[1][1] + dy), op[2], op[3])
    if k == "text":
        return (k, (op[1][0] + dx, op[1][1] + dy), op[2], op[3], op[4], op[5])
    return op


# =============================================================================
# 第 3.5 部分：圆 / 方 —— 依 Andrew 手绘定义的形态表
# =============================================================================
#
# 与三角形是两套不同的东西：
#   三角形  —— 一个图形装一个卦（2^n 种填色）。
#   圆 / 方 —— 数「不管从哪个角度看都不重复」的形态，而形态数本身就是爻位数。
#
#   1 格（整块）  → 2 形态 = 两仪   （阳、阴）
#   2 格（对半）  → 3 形态 = 三爻   （阳=初爻、对半=中爻、阴=上爻）
#   4 格（四分）  → 6 形态 = 六爻   （初爻…上爻）
#
# 每层由左到右 = 阳最盛 → 阴最盛 = 初爻 → 上爻。
# 格号一律：0=左上 1=右上 2=左下 3=右下（2 格时 0=左 1=右；1 格时只有 0）。

LEVEL_NAME = {1: "两仪", 2: "三爻", 3: "六爻"}
LEVEL_CELLS = {1: 1, 2: 2, 3: 4}


def level_info(level, cut="line", order="taohe"):
    """(层名, 格数, 形态数) —— 形态数随分割样式而变。"""
    return (LEVEL_NAME[level], LEVEL_CELLS[level],
            len(form_table(cut, level, order)))

# (形态名, 填阳的格号集合)  —— 顺序即由左至右，阳最盛 → 阴最盛
#
# 直线十字四分 / 四旋螺旋四分：四片彼此全等，转得动 ⇒ 只有 6 个形态。
# 太极（S 线 + 同心圆）四分：内 / 外 是转不掉的分别，只剩「整图转 180°」
#   一个对称 ⇒ 16 种填色收成 10 个形态，不是 6 个。
FORM_TABLE = {
    1: [("阳", (0,)),
        ("阴", ())],
    2: [("初爻", (0, 1)),          # 全填
        ("中爻", (0,)),            # 对半（阳在左）
        ("上爻", ())],             # 全空
    3: [("初爻", (0, 1, 2, 3)),    # 四格全填
        ("二爻", (0, 2, 3)),       # 三格
        ("三爻", (0, 3)),          # 两格·对角
        ("四爻", (2, 3)),          # 两格·相邻
        ("五爻", (2,)),            # 一格
        ("上爻", ())],             # 全空
}

# ---------------------------------------------------------------- 十天干 --
#
# 太极四分恰好有 10 个形态 —— 正好配十天干。
# 索引 0=阳外 1=阳内 2=阴内 3=阴外（阳在左）。
#
# 两种排法都提供，GUI 可切换：
#
# A「阳盛序」—— 沿用你其他各层的规矩：由左至右 = 阳最盛 → 阴最盛，
#   甲配全填、癸配全空，两端对得上阳干之首与阴干之末。
#
# B「五行配对序」—— 把「填 / 空 互换」当作阴阳翻转，十个形态恰好收成 5 组：
#     全填 ↔ 全空 ／ 缺内 ↔ 一内 ／ 缺外 ↔ 一外 ／ 外环 ↔ 内圈
#     一仪、交错 各自互补（自补），两者凑成第五组
#   每组配一个五行，阳干给「阳的那一个」，阴干给「阴的那一个」。
#   一仪 vs 交错 用连 / 断分阴阳（一仪整片相连＝阳，交错只交于一点＝断＝阴），
#   外环 vs 内圈 用外 / 内分阴阳。
TAIJI_FORMS = {
    "全填": (0, 1, 2, 3), "缺内": (0, 1, 3), "缺外": (0, 1, 2),
    "一仪": (0, 1), "交错": (0, 2), "内圈": (1, 2), "外环": (0, 3),
    "一外": (0,), "一内": (1,), "全空": (),
}
TAIJI_DESC = {
    "全填": "四格全填", "缺内": "三格·缺一内", "缺外": "三格·缺一外",
    "一仪": "两格·一仪（相连）", "交错": "两格·交错（只交于一点）",
    "内圈": "两格·内圈", "外环": "两格·外环",
    "一外": "一格·外", "一内": "一格·内", "全空": "全空",
}
TIANGAN = "甲乙丙丁戊己庚辛壬癸"
WUXING = ["木", "木", "火", "火", "土", "土", "金", "金", "水", "水"]

# 「套合序」—— Andrew 的排法（10tiangan.ods），第 i 个与第 11-i 个套起来补成全阳：
#     甲+癸  全填 ∪ 全空
#     乙+壬  缺外 ∪ 一外（壬的实心格转到下面 = 阴外）
#     丙+辛  缺内 ∪ 一内（辛的实心格转到下面 = 阴内）
#     丁+庚  外环 ∪ 内圈（不必转）
#     戊、己 各自自补（交错、一仪 转 180° 即补成全阳）—— 十形态里只有这两个是自补的
# 第三栏是「实际要画的格」，与形态所属的类别可差一个 180°（辛、壬 就是）。
TAIJI_ORDER = {
    "taohe": [("全填", (0, 1, 2, 3)), ("缺外", (0, 1, 2)), ("缺内", (0, 1, 3)),
              ("外环", (0, 3)), ("交错", (0, 2)), ("一仪", (0, 1)),
              ("内圈", (1, 2)), ("一内", (2,)), ("一外", (3,)), ("全空", ())],
    "yang":  [("全填", (0, 1, 2, 3)), ("缺内", (0, 1, 3)), ("缺外", (0, 1, 2)),
              ("一仪", (0, 1)), ("交错", (0, 2)), ("内圈", (1, 2)),
              ("外环", (0, 3)), ("一外", (0,)), ("一内", (1,)), ("全空", ())],
}
TAIJI_L3_ORDER = "taohe"          # 预设 = 你的排法


def taiji_l3(order=None):
    seq = TAIJI_ORDER[order or TAIJI_L3_ORDER]
    return [(TIANGAN[i], cells) for i, (_k, cells) in enumerate(seq)]


def taiji_l3_notes(order=None):
    # 天干本身的五行是固定的（甲乙木、丙丁火…），与排法无关
    seq = TAIJI_ORDER[order or TAIJI_L3_ORDER]
    return [f"{WUXING[i]}　{k}　{TAIJI_DESC[k]}" for i, (k, _c) in enumerate(seq)]


def taohe_pairs(order=None):
    """检查「第 i 个 ∪ 第 11-i 个 = 全阳」，回传 [(甲, 癸, 是否补满, 说明)]。"""
    seq = TAIJI_ORDER[order or TAIJI_L3_ORDER]
    out = []
    n = len(seq)
    for i in range(n // 2):
        j = n - 1 - i
        (ka, ca), (kb, cb) = seq[i], seq[j]
        full = set(ca) | set(cb) == {0, 1, 2, 3}
        disj = not (set(ca) & set(cb))
        if full and disj:
            note = "补满全阳"
        else:
            flip = {0: 3, 1: 2, 2: 1, 3: 0}
            cb2 = {flip[x] for x in cb}
            note = ("转 180° 后补满" if set(ca) | cb2 == {0, 1, 2, 3}
                    and not (set(ca) & cb2) else "补不满")
        out.append((TIANGAN[i], ka, TIANGAN[j], kb, full and disj, note))
    return out


def form_table(cut, level, order=None):
    if cut in TAIJI_LIKE and level == 3:
        return taiji_l3(order)
    return FORM_TABLE[level]


def form_notes(cut, level, order=None):
    if cut in TAIJI_LIKE and level == 3:
        return taiji_l3_notes(order)
    return FORM_NOTE[level]

# 圆可以用直线切，也可以用曲线切；方按手绘只用直线。
# 起始方位：阳片摆哪边。太极预设「阳在上」——甲=乾=1 在上，癸=坤=0 在下。
ORIENTS = {"阳在左": 0, "阳在上": 270, "阴在上": 90, "阳在右": 180}
DEFAULT_ORIENT = {"line": "阳在左", "pinwheel": "阳在左",
                  "taiji": "阳在上", "sqtaiji": "阳在上"}

CUT_STYLES = {
    "line":     "直线（十字）",
    "taiji":    "太极（S 线 + 同心圆）",
    "pinwheel": "四旋（螺旋四臂）",
    "sqtaiji":  "方套太极（外方内太极）",
}
TAIJI_LIKE = ("taiji", "sqtaiji")     # 四分层都是 10 形态、都用 阳外/阳内/阴内/阴外

CELL_LABELS = {
    ("line", 1): ["整块"], ("line", 2): ["左", "右"],
    ("line", 3): ["左上", "右上", "左下", "右下"],
    ("taiji", 1): ["整块"], ("taiji", 2): ["阳仪", "阴仪"],
    ("taiji", 3): ["阳外", "阳内", "阴内", "阴外"],
    ("pinwheel", 1): ["整块"], ("pinwheel", 2): ["阳仪", "阴仪"],
    ("pinwheel", 3): ["左上臂", "右上臂", "左下臂", "右下臂"],
    ("sqtaiji", 1): ["整块"], ("sqtaiji", 2): ["阳半", "阴半"],
    ("sqtaiji", 3): ["阳外", "阳内", "阴内", "阴外"],
}


def cell_labels(cut, level):
    return CELL_LABELS.get((cut, level), CELL_LABELS[("line", level)])


LEVELS = (1, 2, 3)

FORM_NOTE = {
    3: ["四格全填", "三格", "两格·对角", "两格·相邻", "一格", "全空"],
    2: ["全填", "对半", "全空"],
    1: ["全填", "全空"],
}


def _arc(a0, a1, n=120, r=1.0, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
             cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def _clean(pts, tol=1e-9):
    """把顶点吸附到 1e-9 网格并去掉相邻重复点。

    太极四片在 S 线与同心圆的交点上，弧的两种算法会给出相差 1e-16 的
    两个「同一点」，那会让交叉线的穿越计数变成奇数、配对错位而画出穿过
    图形的假线。吸附后角点唯一，半开区间规则就恒给偶数。
    """
    out = []
    for x, y in pts:
        q = (round(x / tol) * tol, round(y / tol) * tol)
        if not out or abs(q[0] - out[-1][0]) > tol / 2 or abs(q[1] - out[-1][1]) > tol / 2:
            out.append(q)
    while len(out) > 1 and abs(out[0][0] - out[-1][0]) <= tol and \
            abs(out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


ROOT_HALF = 1.0 / math.sqrt(2.0)          # 递归太极的内圆半径：面积正好各半


def _rot180(pts):
    return [(-x, -y) for x, y in pts]


def taiji_halves():
    """经典太极：两片面积严格各 π/2。返回 [阳（偏右）, 阴（偏左）]。"""
    yang = (_arc(-math.pi / 2, math.pi / 2, 720) +                 # 外圆右半
            _arc(math.pi / 2, -math.pi / 2, 360, 0.5, 0.0, 0.5) +  # 上半圆，鼓向 +x
            _arc(math.pi / 2, 1.5 * math.pi, 360, 0.5, 0.0, -0.5)) # 下半圆，鼓向 -x
    return [_rot180(yang), yang]      # [阳仪（偏左）, 阴仪（偏右）]


def taiji_quarters():
    """递归太极：太极二分后，各仪再被 r=1/√2 的同心圆切成内外两片。

    因为 S 线关于原点 180° 对称，同心圆被它正好平分，
    所以内片面积 = πρ²/2 = π/4，四片严格相等。
    顺序 [阳外, 阳内, 阴内, 阴外] —— {0,3} 是对位的一对，{2,3} 是相邻的一对，
    与直线版的「对角 / 相邻」读法一致。
    """
    y_out = (_arc(-math.pi / 2, math.pi / 2, 720) +
             _arc(math.pi / 2, 0.0, 180, 0.5, 0.0, 0.5) +
             _arc(math.pi / 4, -0.75 * math.pi, 360, ROOT_HALF) +
             _arc(math.pi, 1.5 * math.pi, 180, 0.5, 0.0, -0.5))
    y_in = (_arc(0.0, -math.pi / 2, 180, 0.5, 0.0, 0.5) +
            _arc(math.pi / 2, math.pi, 180, 0.5, 0.0, -0.5) +
            _arc(1.25 * math.pi, 2.25 * math.pi, 360, ROOT_HALF))
    # 阳在左：把整组转 180°，索引 = [阳外(左), 阳内(左), 阴内(右), 阴外(右)]
    return [_rot180(y_out), _rot180(y_in), y_in, y_out]


SQ_HALF = math.sqrt(2 * math.pi) / 2.0    # 外方半边长：外圈面积 = 内圆面积


def _s_curve(n=360):
    """太极 S 线（圆内），从 (0,1) 到 (0,-1)。"""
    return (_arc(math.pi / 2, -math.pi / 2, n, 0.5, 0.0, 0.5) +
            _arc(math.pi / 2, 1.5 * math.pi, n, 0.5, 0.0, -0.5))


def sq_taiji_halves():
    """外方内太极，二分：S 线上下各接一条竖直线，把整个方形一分为二。

    每半 = 方面积/2。左半为阳（与其他样式一致：阳在左）。
    """
    h = SQ_HALF
    right = ([(0.0, h), (h, h), (h, -h), (0.0, -h), (0.0, -1.0)] +
             _s_curve(360)[::-1] + [(0.0, h)])
    return [_rot180(right), right]


def sq_taiji_quarters():
    """外方内太极，四分：再用圆周分内外。四格面积严格相等 = 方面积/4。

    内两片 = 半圆 = π/2；外两片 = (a²−π)/2 = π/2。顺序 [阳外, 阳内, 阴内, 阴外]。
    """
    h = SQ_HALF
    # 阴内 = 圆的右半（S 线右侧）
    y_in = _arc(-math.pi / 2, math.pi / 2, 360) + _s_curve(360)
    # 阴外 = 方的右半减去圆 —— 外框右半 + 竖线 + 圆的右弧
    y_out = ([(0.0, h), (h, h), (h, -h), (0.0, -h), (0.0, -1.0)] +
             _arc(-math.pi / 2, math.pi / 2, 360) + [(0.0, h)])
    return [_rot180(y_out), _rot180(y_in), y_in, y_out]


def _spiral_arm(th0, twist=math.radians(75), n=160):
    return [((i / n) * math.cos(th0 + twist * i / n),
             (i / n) * math.sin(th0 + twist * i / n)) for i in range(n + 1)]


def pinwheel_quarters(twist=math.radians(75)):
    """四旋太极：四条同形螺旋臂 → 四片彼此全等 ⇒ 面积必然各 π/4。

    环向顺序 c0,c1,c2,c3 → 索引 [c0, c1, c3, c2]，
    使 {0,3} 为对位、{2,3} 为相邻，与直线版读法一致。
    """
    arms = [_spiral_arm(k * math.pi / 2, twist) for k in range(4)]
    cyc = []
    for i in range(4):
        a, b = arms[i], arms[(i + 1) % 4]
        ta = math.atan2(a[-1][1], a[-1][0])
        tb = math.atan2(b[-1][1], b[-1][0])
        while tb <= ta:
            tb += 2 * math.pi
        cyc.append(a + _arc(ta, tb, 180) + b[::-1])
    # 索引对齐正方形：0=左上 1=右上 2=左下 3=右下（按各片质心所在象限）
    # 环向顺序 c0(上偏左) c1(左偏下) c2(下偏右) c3(右偏上)
    # ⇒ [c0, c3, c1, c2]，同时仍保持 {0,3} 对位、{2,3} 相邻
    return [cyc[0], cyc[3], cyc[1], cyc[2]]


def shape_cells(shape: str, level: int, cut: str = "line"):
    """返回该形状该层的格子（每格是一串点，顶点已吸附去重）。"""
    return [_clean(c) for c in _shape_cells_raw(shape, level, cut)]


def _shape_cells_raw(shape: str, level: int, cut: str = "line"):
    if shape == "circle" and cut == "sqtaiji":
        h = SQ_HALF
        if level == 1:
            return [[(-h, -h), (h, -h), (h, h), (-h, h)]]
        return sq_taiji_halves() if level == 2 else sq_taiji_quarters()
    if shape == "circle" and cut != "line":
        if level == 1:
            return [_arc(0, 2 * math.pi, 360)]
        if level == 2:
            return taiji_halves()
        return taiji_quarters() if cut == "taiji" else pinwheel_quarters()
    if shape == "circle":
        if level == 1:
            return [_arc(0, 2 * math.pi, 360)]
        if level == 2:
            return [_arc(math.pi / 2, 1.5 * math.pi, 180),      # 左
                    _arc(-math.pi / 2, math.pi / 2, 180)]       # 右
        q = lambda a0, a1: [(0.0, 0.0)] + _arc(a0, a1, 90)
        return [q(math.pi / 2, math.pi),          # 左上
                q(0.0, math.pi / 2),              # 右上
                q(math.pi, 1.5 * math.pi),        # 左下
                q(1.5 * math.pi, 2 * math.pi)]    # 右下
    if level == 1:
        return [[(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    if level == 2:
        return [[(-1, -1), (0, -1), (0, 1), (-1, 1)],
                [(0, -1), (1, -1), (1, 1), (0, 1)]]
    return [[(-1, 0), (0, 0), (0, 1), (-1, 1)],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(-1, -1), (0, -1), (0, 0), (-1, 0)],
            [(0, -1), (1, -1), (1, 0), (0, 0)]]


def shape_outline(shape: str, cut: str = "line"):
    if cut == "sqtaiji":
        h = SQ_HALF
        return [(-h, -h), (h, -h), (h, h), (-h, h)]
    return (_arc(0, 2 * math.pi, 360) if shape == "circle"
            else [(-1, -1), (1, -1), (1, 1), (-1, 1)])


def shape_dividers(shape: str, level: int, cut: str = "line"):
    """分格线（空形态也要看得见，如你手绘）。每条是一串点。"""
    if level == 1:
        return []
    if cut == "sqtaiji":
        h = SQ_HALF
        spine = [(0.0, h), (0.0, 1.0)] + _s_curve(180) + [(0.0, -h)]
        if level == 2:
            return [spine]
        return [spine, _arc(0, 2 * math.pi, 360)]
    if shape == "circle" and cut != "line":
        s_curve = (_arc(math.pi / 2, -math.pi / 2, 180, 0.5, 0.0, 0.5) +
                   _arc(math.pi / 2, 1.5 * math.pi, 180, 0.5, 0.0, -0.5))
        if level == 2:
            return [s_curve]
        if cut == "taiji":
            return [s_curve, _arc(0, 2 * math.pi, 360, ROOT_HALF)]
        return [_spiral_arm(k * math.pi / 2) for k in range(4)]
    v = [(0.0, 1.0), (0.0, -1.0)]
    h = [(-1.0, 0.0), (1.0, 0.0)]
    return [v] if level == 2 else [v, h]


def shape_dots(shape: str, level: int, cut: str = "line"):
    if level == 1:
        return []
    if cut == "sqtaiji":
        h = SQ_HALF
        d = [(0.0, h), (0.0, -h), (0.0, 1.0), (0.0, -1.0), (0.0, 0.0)]
        return d
    if shape == "circle" and cut != "line":
        d = [(0.0, 1.0), (0.0, -1.0), (0.0, 0.0)]
        return d if level == 2 else d + [(0.5, 0.5), (-0.5, -0.5)]
    if level == 2:
        return [(0.0, 1.0), (0.0, -1.0)]
    return [(0.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0)]


@dataclass
class Form:
    """圆或方的一个形态。"""
    shape: str = "circle"        # circle | square
    cut: str = "line"            # line / taiji / pinwheel（仅圆可换）
    order: str = "taohe"         # 太极四分层的十干排法：taohe / yang
    level: int = 3               # 1 / 2 / 3
    index: int = 0               # 该层第几个（0 起）
    name: str = ""
    filled: tuple = ()           # 填阳的格号
    colors: dict = field(default_factory=dict)   # 格号 -> 自定颜色

    @staticmethod
    def default(shape, level, index, cut="line", order="taohe"):
        c = cut if shape == "circle" else "line"
        nm, fl = form_table(c, level, order)[index]
        return Form(shape, cut, order, level, index, nm, tuple(fl))

    @property
    def note(self):
        c = self.cut if self.shape == "circle" else "line"
        return form_notes(c, self.level, self.order)[self.index]

    @property
    def cells(self):
        return shape_cells(self.shape, self.level, self.cut)

    @property
    def labels(self):
        return cell_labels(self.cut if self.shape == "circle" else "line",
                           self.level)

    def caption(self):
        return f"{self.index+1}  {self.name}"


def build_form_ops(f: Form, style: dict, show_vertices=True, show_grid=True,
                   show_title=True, title=None, caption_note=False,
                   rotation=0):
    """与三角形共用同一套绘图指令，填色风格一致。"""
    st = style
    ops = []
    cut0 = f.cut if f.shape == "circle" else "line"
    cells = f.cells
    ds = math.sqrt(4.0 / max(1, len(cells)))
    ds = max(ds, 0.7)
    gap = max(0.012, st["hatch_spacing"] * ds * 1.35)

    for i, poly in enumerate(cells):
        yang = i in f.filled
        fill = f.colors.get(i) or (st["yang_fill"] if yang else st["yin_fill"])
        ops.append(("poly", poly, fill, None, 0.0))
    for i, poly in enumerate(cells):
        if i in f.filled:
            for p, q in hatch_segments(poly, gap):
                ops.append(("line", p, q, st["hatch_color"],
                            st["hatch_width"] * 1.1))
    if show_grid:
        cut = cut0
        for line in shape_dividers(f.shape, f.level, cut):
            for i in range(len(line) - 1):
                ops.append(("line", line[i], line[i + 1], st["line_color"],
                            st["line_width"] * 1.2))
    ops.append(("poly", shape_outline(f.shape, cut0), None, st["outer_color"],
                st["outer_width"]))
    if show_vertices:
        cut = cut0
        for p in shape_dots(f.shape, f.level, cut):
            ops.append(("dot", p, st["vertex_r"] * 0.95, st["vertex_inner"]))
    ops = rotate_ops(ops, rotation)
    if show_title:
        t = title if title is not None else f.caption()
        if t:
            ops.append(("text", (0.0, 1.30), t, st["title_size"] * 0.95,
                        st["text_color"], "middle"))
    if caption_note:
        ops.append(("text", (0.0, -1.34), f.note,
                    st["font_size"] * 1.15, "#666666", "middle"))
    return ops


def form_bbox(show_title=True, caption_note=False, cut="line"):
    m = SQ_HALF + 0.06 if cut == "sqtaiji" else 1.12
    top = m + 0.33 if show_title else m
    bot = -(m + 0.36) if caption_note else -m
    return (-m, bot, m, top)


def form_to_svg(f: Form, style: dict, px=760, **kw):
    ops = build_form_ops(f, style, **kw)
    cut0 = f.cut if f.shape == "circle" else "line"
    return ops_to_svg(ops, form_bbox(kw.get("show_title", True),
                                     kw.get("caption_note", False), cut0),
                      px, style["bg"], f.name or "form")


def formset_default(shape: str, cut: str = "line", order: str = "taohe"):
    c = cut if shape == "circle" else "line"
    return [Form.default(shape, lv, i, cut, order)
            for lv in LEVELS for i in range(len(form_table(c, lv, order)))]


def formset_to_svg(forms, style, px_each=260, show_vertices=False,
                   show_grid=True, rotation=0):
    """把 2 + 3 + 6 三行拼成一张总图，行内居中。"""
    cut0 = forms[0].cut if forms and forms[0].shape == "circle" else "line"
    bb = form_bbox(True, True, cut0)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    cw, ch = w * 1.12, h * 1.12
    rows = [[f for f in forms if f.level == lv] for lv in LEVELS]
    cols = max(len(r) for r in rows)
    all_ops = []
    for r, row in enumerate(rows):
        off = (cols - len(row)) / 2.0
        for c, f in enumerate(row):
            ox = (off + c) * cw - bb[0]
            oy = (len(rows) - 1 - r) * ch - bb[1]
            for op in build_form_ops(f, style, caption_note=True,
                                     show_vertices=show_vertices,
                                     show_grid=show_grid, rotation=rotation):
                all_ops.append(_shift_op(op, ox, oy))
    total = (0.0, 0.0, cw * cols, ch * len(rows))
    return ops_to_svg(all_ops, total, px_each * cols, style["bg"], "形态表")


# =============================================================================
# 第 4 部分：批量导出（命令行也能用）
# =============================================================================

def batch_export(outdir, n=6, mode="gua", mirror=False, high_first=True,
                 point_down=True, scheme="binary", fmt="svg", px=520,
                 sheet=True, style=None, show_names=True, qt_png=None,
                 progress=None, last_cut="median"):
    os.makedirs(outdir, exist_ok=True)
    codes = order_codes(n, scheme)
    models, written = [], []
    for k, code in enumerate(codes):
        m = Model(n=n, mode=mode, mirror=mirror, high_first=high_first,
                  point_down=point_down, last_cut=last_cut)
        if style:
            m.style.update(style)
        m.show_names = show_names
        m.rebuild(keep_styles=False)
        m.apply_code(code)
        m.title = gua_name(code, n)
        models.append(m)
        base = f"{k+1:02d}_{code_bin(code, n)}_{safe(m.title)}"
        if fmt in ("svg", "both"):
            p = os.path.join(outdir, base + ".svg")
            with open(p, "w", encoding="utf-8") as f:
                f.write(model_to_svg(m, px))
            written.append(p)
        if fmt in ("png", "both") and qt_png:
            p = os.path.join(outdir, base + ".png")
            qt_png(m, p, px)
            written.append(p)
        if progress:
            progress(k + 1, len(codes))

    if sheet:
        cols = 8 if n >= 5 else (4 if n >= 3 else 2)
        p = os.path.join(outdir, f"_sheet_2^{n}_{scheme}.svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(sheet_to_svg(models, cols, px_each=px // 2 + 60))
        written.append(p)
    return written


def code_bin(code, n):
    return "".join(str((code >> i) & 1) for i in range(n - 1, -1, -1))


def safe(s):
    bad = '<>:"/\\|?*'
    return "".join("_" if c in bad else c for c in str(s)).strip() or "x"


# =============================================================================
# 第 5 部分：PyQt6 图形界面
# =============================================================================

def run_gui():
    try:
        from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
        from PyQt6.QtGui import (QPainter, QPolygonF, QColor, QPen, QBrush,
                                 QFont, QImage, QAction, QKeySequence)
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QGridLayout, QLabel, QSpinBox, QComboBox, QCheckBox, QPushButton,
            QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog, QSplitter,
            QGroupBox, QColorDialog, QMessageBox, QScrollArea, QDoubleSpinBox,
            QHeaderView, QTextEdit, QDialog, QDialogButtonBox, QAbstractItemView,
            QTabWidget, QFrame, QSizePolicy)
    except ImportError:
        print("需要 PyQt6：  pip install PyQt6", file=sys.stderr)
        return 2

    # ---------------------------------------------------------- 画布 --
    class Canvas(QWidget):
        cellClicked = pyqtSignal(int, bool)   # (格号, 是否双击)

        def __init__(self, model, parent=None):
            super().__init__(parent)
            self.m = model
            self.selected = -1
            self.setMinimumSize(420, 420)
            self.setMouseTracking(True)

        # 模型坐标 → 屏幕坐标
        def _tf(self):
            x0, y0, x1, y1 = self.m.bbox()
            w, h = x1 - x0, y1 - y0
            W, H = self.width(), self.height()
            s = min(W / w, H / h) * 0.96
            ox = (W - w * s) / 2 - x0 * s
            oy = (H - h * s) / 2 + y1 * s
            return s, ox, oy

        def paintEvent(self, ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.fillRect(self.rect(), QColor(self.m.style["bg"]))
            s, ox, oy = self._tf()
            paint_ops(p, build_ops(self.m), s, ox, oy)
            if 0 <= self.selected < len(self.m.layout.cells):
                t = self.m.layout.cells[self.selected]
                poly = QPolygonF(
                    [QPointF(ox + q[0] * s, oy - q[1] * s)
                     for q in (rot_pt(v, self.m.rotation) for v in t.pts)])
                pen = QPen(QColor("#ff6d00"))
                pen.setWidthF(max(1.6, 0.016 * s))
                p.setPen(pen)
                p.setBrush(QBrush(QColor(255, 109, 0, 46)))
                p.drawPolygon(poly)
            p.end()

        def _hit(self, pos):
            s, ox, oy = self._tf()
            mx = (pos.x() - ox) / s
            my = (oy - pos.y()) / s
            return self.m.layout.cell_at(rot_pt((mx, my), -self.m.rotation))

        def mousePressEvent(self, ev):
            i = self._hit(ev.position())
            if i >= 0:
                self.selected = i
                self.cellClicked.emit(i, False)
                self.update()

        def mouseDoubleClickEvent(self, ev):
            i = self._hit(ev.position())
            if i >= 0:
                self.selected = i
                self.cellClicked.emit(i, True)
                self.update()

    def paint_ops(p, ops, s, ox, oy):
        def P(q):
            return QPointF(ox + q[0] * s, oy - q[1] * s)
        for op in ops:
            k = op[0]
            if k == "poly":
                _, pts, fill, stroke, lw = op
                poly = QPolygonF([P(q) for q in pts])
                p.setBrush(QBrush(QColor(fill)) if fill else Qt.BrushStyle.NoBrush)
                if stroke:
                    pen = QPen(QColor(stroke)); pen.setWidthF(max(0.7, lw * s))
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin); p.setPen(pen)
                else:
                    p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(poly)
            elif k == "line":
                _, a, b, col, lw = op
                pen = QPen(QColor(col)); pen.setWidthF(max(0.6, lw * s))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen); p.drawLine(P(a), P(b))
            elif k == "dot":
                _, q, r, col = op
                p.setPen(QPen(QColor("#ffffff"), max(0.5, 0.004 * s)))
                p.setBrush(QBrush(QColor(col)))
                p.drawEllipse(P(q), r * s, r * s)
            elif k == "text":
                _, q, txt, size, col, _anchor = op
                f = QFont(); f.setPixelSize(max(7, int(size * s)))
                p.setFont(f); p.setPen(QPen(QColor(col)))
                pt = P(q)
                rect = QRectF(pt.x() - 6 * size * s, pt.y() - size * s,
                              12 * size * s, 2 * size * s)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)

    def _paint_bbox(painter, ops, bbox, W, H, margin=0.97):
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        s = min(W / w, H / h) * margin
        ox = (W - w * s) / 2 - x0 * s
        oy = (H - h * s) / 2 + y1 * s
        paint_ops(painter, ops, s, ox, oy)

    def render_png(m: Model, path, px=900):
        x0, y0, x1, y1 = m.bbox()
        w, h = x1 - x0, y1 - y0
        s = px / max(w, h)
        W, H = int(round(w * s)), int(round(h * s))
        img = QImage(W, H, QImage.Format.Format_ARGB32)
        img.fill(QColor(m.style["bg"]))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        paint_ops(p, build_ops(m), s, -x0 * s, y1 * s)
        p.end()
        img.save(path)

    def render_ops_png(ops, bbox, path, style, px=900):
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        s = px / max(w, h)
        W, H = int(round(w * s)), int(round(h * s))
        img = QImage(W, H, QImage.Format.Format_ARGB32)
        img.fill(QColor(style["bg"]))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        paint_ops(p, ops, s, -x0 * s, y1 * s)
        p.end()
        img.save(path)


    class TriangleTab(QWidget):
        def __init__(self, status):
            super().__init__()
            self.status = status
            self.m = Model(n=3)
            self.m.rebuild(keep_styles=False)
            self._loading = False

            self.canvas = Canvas(self.m)
            self.canvas.cellClicked.connect(self.on_cell_clicked)

            panel = QWidget()
            pl = QVBoxLayout(panel)
            pl.setContentsMargins(8, 8, 8, 8)

            # --- 结构 ---
            g1 = QGroupBox("结构")
            f1 = QGridLayout(g1)
            self.cb_n = QComboBox()
            for _n in GUA_LEVELS:
                self.cb_n.addItem(f"{_n}　{GUA_LEVEL_NAMES[_n]}", _n)
            self.cb_n.setCurrentIndex(GUA_LEVELS.index(3))
            self.lb_info = QLabel()
            self.cb_mode = QComboBox()
            self.cb_mode.addItems(["卦形（按爻分区）", "均分（纯等积，自由填色）"])
            f1.addWidget(QLabel("层数 n"), 0, 0); f1.addWidget(self.cb_n, 0, 1)
            f1.addWidget(self.lb_info, 0, 2, 1, 2)
            f1.addWidget(QLabel("模式"), 1, 0); f1.addWidget(self.cb_mode, 1, 1, 1, 3)
            self.ck_high = QCheckBox("大块归上爻（取消 = 归初爻）"); self.ck_high.setChecked(True)
            self.ck_mirror = QCheckBox("左右镜像")
            self.ck_down = QCheckBox("尖朝下（倒三角）"); self.ck_down.setChecked(True)
            f1.addWidget(self.ck_high, 2, 0, 1, 4)
            f1.addWidget(self.ck_mirror, 3, 0, 1, 2)
            f1.addWidget(self.ck_down, 3, 2, 1, 2)
            rot = QHBoxLayout()
            self.btn_ccw = QPushButton("⟲ 逆时针 90°")
            self.btn_cw = QPushButton("⟳ 顺时针 90°")
            self.lb_rot = QLabel("0°")
            rot.addWidget(self.btn_ccw); rot.addWidget(self.btn_cw)
            rot.addWidget(self.lb_rot)
            f1.addLayout(rot, 5, 0, 1, 4)
            self.cb_cut = QComboBox()
            self.cb_cut.addItem("中线切（与原 GeoGebra 一致）", "median")
            self.cb_cut.addItem("斜切（破镜像对称，2ⁿ 个图形全不全等）", "oblique")
            f1.addWidget(QLabel("末两爻切法"), 4, 0)
            f1.addWidget(self.cb_cut, 4, 1, 1, 3)
            pl.addWidget(g1)

            # --- 卦 ---
            self.g2 = QGroupBox("卦")
            f2 = QVBoxLayout(self.g2)
            self.cb_gua = QComboBox()
            f2.addWidget(self.cb_gua)
            self.yao_box = QWidget(); self.yao_lay = QVBoxLayout(self.yao_box)
            self.yao_lay.setContentsMargins(0, 4, 0, 0); self.yao_lay.setSpacing(3)
            f2.addWidget(self.yao_box)
            row = QHBoxLayout()
            self.ed_title = QLineEdit()
            row.addWidget(QLabel("标题")); row.addWidget(self.ed_title)
            f2.addLayout(row)
            pl.addWidget(self.g2)

            # --- 外观 ---
            g3 = QGroupBox("外观")
            f3 = QGridLayout(g3)
            self.btn_yang = QPushButton("阳底色"); self.btn_yin = QPushButton("阴底色")
            self.btn_hatch = QPushButton("交叉线色"); self.btn_line = QPushButton("格线色")
            f3.addWidget(self.btn_yang, 0, 0); f3.addWidget(self.btn_yin, 0, 1)
            f3.addWidget(self.btn_hatch, 0, 2); f3.addWidget(self.btn_line, 0, 3)
            self.sp_gap = QDoubleSpinBox(); self.sp_gap.setRange(0.01, 0.30)
            self.sp_gap.setSingleStep(0.005); self.sp_gap.setDecimals(3)
            self.sp_gap.setValue(DEFAULT_STYLE["hatch_spacing"])
            f3.addWidget(QLabel("交叉线间距"), 1, 0); f3.addWidget(self.sp_gap, 1, 1)
            self.ck_auto = QCheckBox("细节随格子大小自动缩放")
            self.ck_auto.setChecked(True)
            f3.addWidget(self.ck_auto, 3, 0, 1, 4)
            self.ck_v = QCheckBox("顶点"); self.ck_v.setChecked(True)
            self.ck_g = QCheckBox("格线"); self.ck_g.setChecked(True)
            self.ck_i = QCheckBox("编号")
            self.ck_nm = QCheckBox("名称"); self.ck_nm.setChecked(True)
            self.ck_t = QCheckBox("标题"); self.ck_t.setChecked(True)
            f3.addWidget(self.ck_v, 2, 0); f3.addWidget(self.ck_g, 2, 1)
            f3.addWidget(self.ck_i, 2, 2); f3.addWidget(self.ck_nm, 2, 3)
            f3.addWidget(self.ck_t, 1, 2, 1, 2)
            pl.addWidget(g3)

            # --- 格子表 ---
            g4 = QGroupBox("格子（双击画布切换阴阳；表内可命名、改色）")
            f4 = QVBoxLayout(g4)
            self.tbl = QTableWidget(0, 5)
            self.tbl.setHorizontalHeaderLabels(["#", "爻位", "阴阳", "名称", "颜色"])
            self.tbl.verticalHeader().setVisible(False)
            self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            hh = self.tbl.horizontalHeader()
            hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            self.tbl.setColumnWidth(0, 34); self.tbl.setColumnWidth(1, 58)
            self.tbl.setColumnWidth(2, 50); self.tbl.setColumnWidth(4, 66)
            f4.addWidget(self.tbl)
            hb = QHBoxLayout()
            self.btn_auto = QPushButton("按爻位自动命名")
            self.btn_clear = QPushButton("清空名称")
            hb.addWidget(self.btn_auto); hb.addWidget(self.btn_clear)
            f4.addLayout(hb)
            pl.addWidget(g4, 1)

            # --- 输出 ---
            g5 = QGroupBox("输出")
            f5 = QGridLayout(g5)
            self.btn_svg = QPushButton("导出 SVG")
            self.btn_png = QPushButton("导出 PNG")
            self.btn_save = QPushButton("保存 JSON")
            self.btn_load = QPushButton("读取 JSON")
            self.btn_batch = QPushButton("批量导出全部 2ⁿ")
            self.btn_check = QPushButton("几何校验")
            f5.addWidget(self.btn_svg, 0, 0); f5.addWidget(self.btn_png, 0, 1)
            f5.addWidget(self.btn_save, 1, 0); f5.addWidget(self.btn_load, 1, 1)
            f5.addWidget(self.btn_batch, 2, 0); f5.addWidget(self.btn_check, 2, 1)
            pl.addWidget(g5)

            sc = QScrollArea(); sc.setWidget(panel); sc.setWidgetResizable(True)
            sc.setMinimumWidth(430); sc.setMaximumWidth(520)
            sp = QSplitter(); sp.addWidget(self.canvas); sp.addWidget(sc)
            sp.setStretchFactor(0, 1)
            _lay = QVBoxLayout(self)
            _lay.setContentsMargins(0, 0, 0, 0)
            _lay.addWidget(sp)

            # 连接
            self.cb_n.currentIndexChanged.connect(self.on_struct)
            self.cb_mode.currentIndexChanged.connect(self.on_struct)
            for ck in (self.ck_high, self.ck_mirror, self.ck_down):
                ck.toggled.connect(self.on_struct)
            self.cb_cut.currentIndexChanged.connect(self.on_struct)
            self.btn_ccw.clicked.connect(lambda: self.rotate(90))
            self.btn_cw.clicked.connect(lambda: self.rotate(-90))
            self.cb_gua.currentIndexChanged.connect(self.on_gua_combo)
            self.ed_title.textEdited.connect(self.on_title)
            for ck, at in ((self.ck_v, "show_vertices"), (self.ck_g, "show_grid"),
                           (self.ck_i, "show_index"), (self.ck_nm, "show_names"),
                           (self.ck_t, "show_title")):
                ck.toggled.connect(lambda v, a=at: self.set_show(a, v))
            self.sp_gap.valueChanged.connect(self.on_gap)
            self.ck_auto.toggled.connect(self.on_auto)
            self.btn_yang.clicked.connect(lambda: self.pick("yang_fill"))
            self.btn_yin.clicked.connect(lambda: self.pick("yin_fill"))
            self.btn_hatch.clicked.connect(lambda: self.pick("hatch_color"))
            self.btn_line.clicked.connect(lambda: self.pick("line_color"))
            self.tbl.cellChanged.connect(self.on_table_edit)
            self.tbl.cellClicked.connect(self.on_table_click)
            self.btn_auto.clicked.connect(self.auto_name)
            self.btn_clear.clicked.connect(self.clear_names)
            self.btn_svg.clicked.connect(self.export_svg)
            self.btn_png.clicked.connect(self.export_png)
            self.btn_save.clicked.connect(self.save_json)
            self.btn_load.clicked.connect(self.load_json)
            self.btn_batch.clicked.connect(self.batch)
            self.btn_check.clicked.connect(self.check)

            self.rebuild_all()

        # ---------------------------------------------------- helpers --
        def set_show(self, attr, v):
            setattr(self.m, attr, v); self.canvas.update()

        def rotate(self, d):
            self.m.rotation = (self.m.rotation + d) % 360
            self.lb_rot.setText(f"{self.m.rotation}°")
            self.canvas.update()
            self.status(f"旋转 {self.m.rotation}°")

        def on_gap(self, v):
            self.m.style["hatch_spacing"] = float(v); self.canvas.update()

        def on_auto(self, v):
            self.m.style["auto_detail"] = bool(v); self.canvas.update()

        def pick(self, key):
            c = QColorDialog.getColor(QColor(self.m.style[key]), self, "选择颜色")
            if c.isValid():
                self.m.style[key] = c.name(); self.canvas.update()

        def on_struct(self, *_):
            if self._loading:
                return
            self.m.n = self.cb_n.currentData() or 3
            self.m.mode = "gua" if self.cb_mode.currentIndex() == 0 else "even"
            self.m.high_first = self.ck_high.isChecked()
            self.m.mirror = self.ck_mirror.isChecked()
            self.m.point_down = self.ck_down.isChecked()
            self.m.last_cut = self.cb_cut.currentData() or "median"
            self.m.code &= (1 << self.m.n) - 1
            self.m.title = ""
            self.m.rebuild(keep_styles=False)
            self.rebuild_all()

        def rebuild_all(self):
            self._loading = True
            if self.m.n not in GUA_LEVELS:      # 四爻、五爻不存在，任何来路都挡掉
                self.m.n = 3
                self.m.rebuild(keep_styles=False)
            self.cb_n.setCurrentIndex(GUA_LEVELS.index(self.m.n))
            n, N = self.m.n, self.m.layout.n_cells
            self.lb_info.setText(f"2^{n} = {2**n} 种组合 · {N} 个等积格")
            self.g2.setEnabled(self.m.mode == "gua")
            # 卦下拉
            self.cb_gua.clear()
            for c in range(2 ** n):
                sym = gua_symbol(c, n)
                self.cb_gua.addItem(
                    f"{code_bin(c,n)}  {code_string(c,n)}  "
                    f"{gua_name(c,n)}{('  '+sym) if sym else ''}", c)
            idx = self.cb_gua.findData(self.m.code)
            self.cb_gua.setCurrentIndex(max(0, idx))
            # 爻按钮（上爻在最上）
            while self.yao_lay.count():
                w = self.yao_lay.takeAt(0).widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()
            self.yao_btns = {}
            for li in range(n - 1, -1, -1):
                b = QPushButton(); b.setCheckable(True)
                b.clicked.connect(lambda _c, i=li: self.toggle_yao(i))
                self.yao_lay.addWidget(b)
                self.yao_btns[li] = b
            self.ed_title.setText(self.m.title)
            self._loading = False
            self.refresh_yao_buttons()
            self.refresh_table()
            self.canvas.selected = -1
            self.canvas.update()

        def refresh_yao_buttons(self):
            if self.m.mode != "gua":
                return
            b = bits_of(self.m.code, self.m.n)
            for li, btn in self.yao_btns.items():
                yang = bool(b[li])
                btn.setChecked(yang)
                cnt = len(self.m.layout.lines[li])
                btn.setText(f"{self.m.layout.yao_label(li)}   "
                            f"{'▅▅▅ 阳' if yang else '▅ ▅ 阴'}   ({cnt} 格)")

        def toggle_yao(self, li):
            self.m.apply_code(self.m.code ^ (1 << li))
            self._loading = True
            i = self.cb_gua.findData(self.m.code)
            self.cb_gua.setCurrentIndex(max(0, i))
            self.ed_title.setText(self.m.title)
            self._loading = False
            self.refresh_yao_buttons(); self.refresh_table(); self.canvas.update()

        def on_gua_combo(self, _i):
            if self._loading or self.m.mode != "gua":
                return
            code = self.cb_gua.currentData()
            if code is None:
                return
            self.m.title = ""
            self.m.apply_code(int(code))
            self._loading = True
            self.ed_title.setText(self.m.title)
            self._loading = False
            self.refresh_yao_buttons(); self.refresh_table(); self.canvas.update()

        def on_title(self, s):
            self.m.title = s; self.canvas.update()

        # ------------------------------------------------------ 表格 --
        def refresh_table(self):
            self._loading = True
            self.tbl.setRowCount(self.m.layout.n_cells)
            for i in range(self.m.layout.n_cells):
                cs = self.m.cells[i]
                it = QTableWidgetItem(str(i))
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.tbl.setItem(i, 0, it)
                it = QTableWidgetItem(self.m.layout.yao_label(self.m.layout.cell_line[i]))
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.tbl.setItem(i, 1, it)
                it = QTableWidgetItem("阳" if cs.yang else "阴")
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if cs.yang
                                 else Qt.CheckState.Unchecked)
                self.tbl.setItem(i, 2, it)
                self.tbl.setItem(i, 3, QTableWidgetItem(cs.name))
                it = QTableWidgetItem(cs.fill or "默认")
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                it.setBackground(QBrush(QColor(self.m.cell_fill(i))))
                self.tbl.setItem(i, 4, it)
            self._loading = False

        def on_table_edit(self, r, c):
            if self._loading or r >= len(self.m.cells):
                return
            if c == 2:
                self.m.cells[r].yang = (self.tbl.item(r, 2).checkState()
                                        == Qt.CheckState.Checked)
                self.tbl.item(r, 2).setText("阳" if self.m.cells[r].yang else "阴")
                self._loading = True
                self.tbl.item(r, 4).setBackground(QBrush(QColor(self.m.cell_fill(r))))
                self._loading = False
            elif c == 3:
                self.m.cells[r].name = self.tbl.item(r, 3).text()
            self.canvas.update()

        def on_table_click(self, r, c):
            self.canvas.selected = r
            self.canvas.update()
            if c == 4:
                cur = QColor(self.m.cell_fill(r))
                col = QColorDialog.getColor(cur, self, f"第 {r} 格底色")
                if col.isValid():
                    self.m.cells[r].fill = col.name()
                    self.refresh_table(); self.canvas.update()

        def on_cell_clicked(self, i, dbl):
            if dbl:
                self.m.cells[i].yang = not self.m.cells[i].yang
                if self.m.mode == "gua":
                    # 同爻的格子一起翻，保持卦码自洽
                    li = self.m.layout.cell_line[i]
                    v = self.m.cells[i].yang
                    for j in self.m.layout.lines[li]:
                        self.m.cells[j].yang = v
                    code = 0
                    for l2 in range(self.m.n):
                        if self.m.cells[self.m.layout.lines[l2][0]].yang:
                            code |= (1 << l2)
                    self.m.title = ""
                    self.m.apply_code(code)
                    self._loading = True
                    k = self.cb_gua.findData(self.m.code)
                    self.cb_gua.setCurrentIndex(max(0, k))
                    self.ed_title.setText(self.m.title)
                    self._loading = False
                    self.refresh_yao_buttons()
                self.refresh_table()
            self.tbl.selectRow(i)
            self.status(
                f"第 {i} 格 · {self.m.layout.yao_label(self.m.layout.cell_line[i])}"
                f" · 面积 {self.m.layout.cells[i].area:.6g}")
            self.canvas.update()

        def auto_name(self):
            cnt = {}
            for i in range(self.m.layout.n_cells):
                li = self.m.layout.cell_line[i]
                lab = self.m.layout.yao_label(li) if li >= 0 else "格"
                cnt[lab] = cnt.get(lab, 0) + 1
                n_in = len(self.m.layout.lines[li]) if li >= 0 else 0
                self.m.cells[i].name = lab if n_in == 1 else f"{lab}{cnt[lab]}"
            self.refresh_table(); self.canvas.update()

        def clear_names(self):
            for c in self.m.cells:
                c.name = ""
            self.refresh_table(); self.canvas.update()

        # ------------------------------------------------------ 输出 --
        def export_svg(self):
            p, _ = QFileDialog.getSaveFileName(
                self, "导出 SVG", f"{safe(self.m.title or 'guaxing')}.svg",
                "SVG (*.svg)")
            if p:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(model_to_svg(self.m, 1000))
                self.status("已导出 " + p)

        def export_png(self):
            p, _ = QFileDialog.getSaveFileName(
                self, "导出 PNG", f"{safe(self.m.title or 'guaxing')}.png",
                "PNG (*.png)")
            if p:
                render_png(self.m, p, 1400)
                self.status("已导出 " + p)

        def save_json(self):
            p, _ = QFileDialog.getSaveFileName(
                self, "保存 JSON", f"{safe(self.m.title or 'guaxing')}.json",
                "JSON (*.json)")
            if p:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(self.m.to_dict(), f, ensure_ascii=False, indent=2)
                self.status("已保存 " + p)

        def load_json(self):
            p, _ = QFileDialog.getOpenFileName(self, "读取 JSON", "", "JSON (*.json)")
            if not p:
                return
            try:
                with open(p, encoding="utf-8") as f:
                    self.m = Model.from_dict(json.load(f))
            except Exception as e:
                QMessageBox.warning(self, "读取失败", str(e)); return
            self.canvas.m = self.m
            self._loading = True
            self.cb_mode.setCurrentIndex(0 if self.m.mode == "gua" else 1)
            self.ck_high.setChecked(self.m.high_first)
            self.ck_mirror.setChecked(self.m.mirror)
            self.ck_down.setChecked(self.m.point_down)
            self.cb_cut.setCurrentIndex(
                max(0, self.cb_cut.findData(self.m.last_cut)))
            self.ck_v.setChecked(self.m.show_vertices)
            self.ck_g.setChecked(self.m.show_grid)
            self.ck_i.setChecked(self.m.show_index)
            self.ck_nm.setChecked(self.m.show_names)
            self.ck_t.setChecked(self.m.show_title)
            self.sp_gap.setValue(self.m.style.get("hatch_spacing", 0.055))
            self.ck_auto.setChecked(self.m.style.get("auto_detail", True))
            self.lb_rot.setText(f"{self.m.rotation}°")
            self._loading = False
            self.rebuild_all()
            self.status("已读取 " + p)

        def batch(self):
            d = BatchDialog(self.m, self)
            if d.exec() != QDialog.DialogCode.Accepted:
                return
            outdir, scheme, fmt, px, sheet = d.values()
            if not outdir:
                return
            try:
                files = batch_export(
                    outdir, n=self.m.n, mode=self.m.mode, mirror=self.m.mirror,
                    high_first=self.m.high_first, point_down=self.m.point_down,
                    scheme=scheme, fmt=fmt, px=px, sheet=sheet,
                    style=self.m.style, show_names=False,
                    last_cut=self.m.last_cut,
                    qt_png=(lambda mm, pp, xx: render_png(mm, pp, xx)))
            except Exception as e:
                QMessageBox.warning(self, "批量导出失败", str(e)); return
            QMessageBox.information(
                self, "完成", f"已写出 {len(files)} 个文件到\n{outdir}")

        def check(self):
            ok, msgs = validate_layout(self.m.layout)
            dlg = QDialog(self); dlg.setWindowTitle("几何校验")
            v = QVBoxLayout(dlg)
            te = QTextEdit(); te.setReadOnly(True)
            head = ("全部通过 ✓" if ok else "有问题 ✗") + "\n\n"
            te.setPlainText(head + "\n".join(msgs))
            te.setMinimumSize(560, 380)
            v.addWidget(te)
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            bb.accepted.connect(dlg.accept); v.addWidget(bb)
            dlg.exec()

    class BatchDialog(QDialog):
        def __init__(self, m, parent=None):
            super().__init__(parent)
            self.setWindowTitle(f"批量导出全部 2^{m.n} = {2**m.n} 张")
            g = QGridLayout(self)
            self.ed_dir = QLineEdit()
            b = QPushButton("浏览…"); b.clicked.connect(self.browse)
            g.addWidget(QLabel("输出文件夹"), 0, 0)
            g.addWidget(self.ed_dir, 0, 1); g.addWidget(b, 0, 2)
            self.cb_scheme = QComboBox()
            self.cb_scheme.addItem("先天二进制序（000…111）", "binary")
            if m.n in (3, 6):
                self.cb_scheme.addItem("先天八卦序（乾兑离震巽坎艮坤）", "xiantian")
                self.cb_scheme.addItem("连山链序（艮巽坎坤兑震离乾）", "lianshan")
            g.addWidget(QLabel("排列顺序"), 1, 0)
            g.addWidget(self.cb_scheme, 1, 1, 1, 2)
            self.cb_fmt = QComboBox(); self.cb_fmt.addItems(["svg", "png", "both"])
            g.addWidget(QLabel("格式"), 2, 0); g.addWidget(self.cb_fmt, 2, 1, 1, 2)
            self.sp_px = QSpinBox(); self.sp_px.setRange(160, 4000); self.sp_px.setValue(520)
            g.addWidget(QLabel("单图边长 px"), 3, 0); g.addWidget(self.sp_px, 3, 1, 1, 2)
            self.ck_sheet = QCheckBox("同时生成一张总图（SVG 拼版）")
            self.ck_sheet.setChecked(True)
            g.addWidget(self.ck_sheet, 4, 0, 1, 3)
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                  QDialogButtonBox.StandardButton.Cancel)
            bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
            g.addWidget(bb, 5, 0, 1, 3)

        def browse(self):
            d = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
            if d:
                self.ed_dir.setText(d)

        def values(self):
            return (self.ed_dir.text().strip(), self.cb_scheme.currentData(),
                    self.cb_fmt.currentText(), self.sp_px.value(),
                    self.ck_sheet.isChecked())


    # =================================================== 圆 / 方 形态表 --
    class FormThumb(QFrame):
        """一个形态的缩略图，可点选。"""
        picked = pyqtSignal(object)

        def __init__(self, form, style, owner, size=150, parent=None):
            super().__init__(parent)
            self.form = form
            self.style = style
            self.owner = owner            # FormTab，取 rotation
            self.selected = False
            self.cap = max(26, int(size * 0.23))
            self.setFixedSize(size, size + self.cap)
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        def paintEvent(self, ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            bg = QColor("#fff4e6") if self.selected else QColor(self.style["bg"])
            p.fillRect(self.rect(), bg)
            pen = QPen(QColor("#ff6d00" if self.selected else "#d7dde5"))
            pen.setWidth(3 if self.selected else 1)
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self.rect().adjusted(1, 1, -2, -2))
            ops = build_form_ops(self.form, self.style, show_title=False,
                                 show_vertices=False,
                                 rotation=self.owner.rotation)
            cut0 = self.form.cut if self.form.shape == "circle" else "line"
            cap = self.cap
            _paint_bbox(p, ops, form_bbox(False, False, cut0),
                        self.width(), self.height() - cap, 0.88)
            f = QFont(); f.setPixelSize(max(10, int(cap * 0.44))); p.setFont(f)
            p.setPen(QPen(QColor("#222222")))
            p.drawText(QRectF(0, self.height() - cap, self.width(), cap * 0.52),
                       Qt.AlignmentFlag.AlignCenter, self.form.caption())
            f.setPixelSize(max(8, int(cap * 0.33))); p.setFont(f)
            p.setPen(QPen(QColor("#777777")))
            p.drawText(QRectF(0, self.height() - cap * 0.5, self.width(), cap * 0.5),
                       Qt.AlignmentFlag.AlignCenter, self.form.note)
            p.end()

        def mousePressEvent(self, ev):
            self.picked.emit(self)

    class FormPreview(QWidget):
        def __init__(self, style, owner, parent=None):
            super().__init__(parent)
            self.style = style
            self.owner = owner
            self.form = None
            self.show_vertices = True
            self.show_grid = True
            self.setMinimumSize(300, 320)

        def paintEvent(self, ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.fillRect(self.rect(), QColor(self.style["bg"]))
            if self.form is not None:
                ops = build_form_ops(self.form, self.style,
                                     show_vertices=self.show_vertices,
                                     show_grid=self.show_grid,
                                     caption_note=True,
                                     rotation=self.owner.rotation)
                cut0 = self.form.cut if self.form.shape == "circle" else "line"
                _paint_bbox(p, ops, form_bbox(True, True, cut0),
                            self.width(), self.height())
            p.end()

    class FormTab(QWidget):
        """圆 / 方：三层形态同时列出，点一个即选中。"""

        def __init__(self, shape, status, parent=None):
            super().__init__(parent)
            self.shape = shape
            self.status = status
            self.rotation = 0
            self.cut = "line"
            self.order = "taohe"
            self.style = dict(DEFAULT_STYLE)
            self.style["hatch_spacing"] = 0.075
            self.forms = formset_default(shape, self.cut, self.order)
            self.cur = None
            self.thumbs = []
            self._loading = False

            # ---- 左：三行形态 ----
            gal = QWidget()
            gl = QVBoxLayout(gal)
            gl.setContentsMargins(10, 10, 10, 10)
            shp = "圆" if shape == "circle" else "方"
            head = QLabel(
                f"{shp}：每层「不管从哪个角度看都不重复」的形态，"
                f"形态数本身就是爻位数。由左至右 = 阳最盛 → 阴最盛 = 初爻 → 上爻。")
            head.setWordWrap(True)
            head.setStyleSheet("color:#555;")
            gl.addWidget(head)

            bar = QHBoxLayout()
            self.cb_cut = QComboBox()
            for k, v in CUT_STYLES.items():
                self.cb_cut.addItem(v, k)
            self.cb_order = QComboBox()
            self.cb_order.addItem("十干·套合序（你的排法）", "taohe")
            self.cb_order.addItem("十干·阳盛序", "yang")
            if shape == "circle":
                bar.addWidget(QLabel("分割样式"))
                bar.addWidget(self.cb_cut)
                bar.addSpacing(10)
                bar.addWidget(self.cb_order)
            else:
                self.cb_cut.setCurrentIndex(0)
                self.cb_cut.hide()
                self.cb_order.hide()
            bar.addSpacing(16)
            self.cb_orient = QComboBox()
            for k in ORIENTS:
                self.cb_orient.addItem(k, k)
            bar.addWidget(QLabel("方位")); bar.addWidget(self.cb_orient)
            bar.addSpacing(10)
            self.btn_ccw = QPushButton("⟲ 逆时针 90°")
            self.btn_cw = QPushButton("⟳ 顺时针 90°")
            self.lb_rot = QLabel("0°")
            bar.addWidget(self.btn_ccw); bar.addWidget(self.btn_cw)
            bar.addWidget(self.lb_rot); bar.addStretch(1)
            gl.addLayout(bar)

            self.rows, self.boxes = {}, {}
            for lv in LEVELS:
                box = QGroupBox()
                hb = QHBoxLayout(box)
                hb.setSpacing(8)
                self.rows[lv] = hb
                self.boxes[lv] = box
                gl.addWidget(box)
            gl.addStretch(1)
            self.build_gallery()
            sc_gal = QScrollArea(); sc_gal.setWidget(gal); sc_gal.setWidgetResizable(True)

            # ---- 右：选中形态 ----
            panel = QWidget()
            pl = QVBoxLayout(panel)
            pl.setContentsMargins(8, 8, 8, 8)
            self.prev = FormPreview(self.style, self)
            self.prev.show_vertices = False
            pl.addWidget(self.prev)

            g1 = QGroupBox("选中的形态")
            f1 = QGridLayout(g1)
            self.lb_sel = QLabel("（点左边任一形态）")
            f1.addWidget(self.lb_sel, 0, 0, 1, 2)
            self.ed_name = QLineEdit()
            f1.addWidget(QLabel("名称"), 1, 0); f1.addWidget(self.ed_name, 1, 1)
            self.cell_box = QWidget()
            self.cell_lay = QGridLayout(self.cell_box)
            self.cell_lay.setContentsMargins(0, 0, 0, 0)
            f1.addWidget(QLabel("填阳的格"), 2, 0)
            f1.addWidget(self.cell_box, 2, 1)
            self.btn_reset = QPushButton("恢复本形态默认")
            f1.addWidget(self.btn_reset, 3, 0, 1, 2)
            pl.addWidget(g1)

            g2 = QGroupBox("外观（本页共用）")
            f2 = QGridLayout(g2)
            self.btn_yang = QPushButton("阳底色"); self.btn_yin = QPushButton("阴底色")
            self.btn_hatch = QPushButton("交叉线色"); self.btn_line = QPushButton("格线色")
            f2.addWidget(self.btn_yang, 0, 0); f2.addWidget(self.btn_yin, 0, 1)
            f2.addWidget(self.btn_hatch, 1, 0); f2.addWidget(self.btn_line, 1, 1)
            self.sp_gap = QDoubleSpinBox(); self.sp_gap.setRange(0.02, 0.30)
            self.sp_gap.setSingleStep(0.005); self.sp_gap.setDecimals(3)
            self.sp_gap.setValue(self.style["hatch_spacing"])
            f2.addWidget(QLabel("交叉线间距"), 2, 0); f2.addWidget(self.sp_gap, 2, 1)
            self.ck_v = QCheckBox("顶点"); self.ck_v.setChecked(False)
            self.ck_g = QCheckBox("格线"); self.ck_g.setChecked(True)
            f2.addWidget(self.ck_v, 3, 0); f2.addWidget(self.ck_g, 3, 1)
            pl.addWidget(g2)

            g3 = QGroupBox("输出")
            f3 = QGridLayout(g3)
            self.btn_svg = QPushButton("本形态 SVG")
            self.btn_png = QPushButton("本形态 PNG")
            self.btn_sheet = QPushButton("整张形态表 SVG")
            self.btn_sheet_png = QPushButton("整张形态表 PNG")
            self.btn_save = QPushButton("保存 JSON")
            self.btn_load = QPushButton("读取 JSON")
            self.btn_taohe = QPushButton("套合校验（首尾补成全阳）")
            f3.addWidget(self.btn_svg, 0, 0); f3.addWidget(self.btn_png, 0, 1)
            f3.addWidget(self.btn_sheet, 1, 0); f3.addWidget(self.btn_sheet_png, 1, 1)
            f3.addWidget(self.btn_save, 2, 0); f3.addWidget(self.btn_load, 2, 1)
            f3.addWidget(self.btn_taohe, 3, 0, 1, 2)
            pl.addWidget(g3)
            pl.addStretch(1)

            sc = QScrollArea(); sc.setWidget(panel); sc.setWidgetResizable(True)
            sc.setMinimumWidth(330); sc.setMaximumWidth(380)
            sp = QSplitter(); sp.addWidget(sc_gal); sp.addWidget(sc)
            sp.setStretchFactor(0, 1)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(sp)

            self.cb_cut.currentIndexChanged.connect(self.on_cut)
            self.cb_order.currentIndexChanged.connect(self.on_order)
            self.cb_orient.currentIndexChanged.connect(self.on_orient)
            self.btn_ccw.clicked.connect(lambda: self.rotate(90))
            self.btn_cw.clicked.connect(lambda: self.rotate(-90))
            self.ed_name.textEdited.connect(self.on_name)
            self.btn_reset.clicked.connect(self.on_reset)
            self.btn_yang.clicked.connect(lambda: self.pick("yang_fill"))
            self.btn_yin.clicked.connect(lambda: self.pick("yin_fill"))
            self.btn_hatch.clicked.connect(lambda: self.pick("hatch_color"))
            self.btn_line.clicked.connect(lambda: self.pick("line_color"))
            self.sp_gap.valueChanged.connect(self.on_gap)
            self.ck_v.toggled.connect(self.on_show)
            self.ck_g.toggled.connect(self.on_show)
            self.btn_svg.clicked.connect(self.exp_svg)
            self.btn_png.clicked.connect(self.exp_png)
            self.btn_sheet.clicked.connect(self.exp_sheet)
            self.btn_sheet_png.clicked.connect(self.exp_sheet_png)
            self.btn_save.clicked.connect(self.save_json)
            self.btn_load.clicked.connect(self.load_json)
            self.btn_taohe.clicked.connect(self.check_taohe)

            self.apply_default_orient()
            self.on_pick(next(t for t in self.thumbs if t.form.level == 3
                              and t.form.index == 0))

        # ----------------------------------------------------------
        def refresh(self):
            for t in self.thumbs:
                t.update()
            self.prev.update()

        def build_gallery(self):
            """按目前的分割样式重建三排缩略图（形态数会随样式改变）。"""
            c = self.cut if self.shape == "circle" else "line"
            self.thumbs = []
            widest = max(len(form_table(c, lv, self.order)) for lv in LEVELS)
            size = 150 if widest <= 6 else (124 if widest <= 8 else 98)
            for lv in LEVELS:
                hb = self.rows[lv]
                hb.setSpacing(4 if size < 120 else 8)
                while hb.count():
                    it = hb.takeAt(0)
                    w = it.widget()
                    if w:
                        w.setParent(None)
                        w.deleteLater()
                nm, ncell, nform = level_info(lv, c)
                self.boxes[lv].setTitle(f"{nm}　·　{ncell} 格　·　{nform} 形态")
                for f in [x for x in self.forms if x.level == lv]:
                    th = FormThumb(f, self.style, self, size)
                    th.picked.connect(self.on_pick)
                    hb.addWidget(th)
                    self.thumbs.append(th)
                hb.addStretch(1)

        def rotate(self, d):
            self.set_rotation((self.rotation + d) % 360)
            self.status(f"旋转 {self.rotation}°")

        def set_rotation(self, deg):
            self.rotation = deg % 360
            self.lb_rot.setText(f"{self.rotation}°")
            self._loading = True
            hit = next((k for k, v in ORIENTS.items() if v == self.rotation), None)
            self.cb_orient.setCurrentIndex(
                self.cb_orient.findData(hit) if hit else -1)
            self._loading = False
            self.refresh()

        def on_orient(self, _i):
            if self._loading:
                return
            k = self.cb_orient.currentData()
            if k:
                self.set_rotation(ORIENTS[k])
                self.status(f"方位：{k}（{self.rotation}°）")

        def apply_default_orient(self):
            c = self.cut if self.shape == "circle" else "line"
            self.set_rotation(ORIENTS[DEFAULT_ORIENT.get(c, "阳在左")])

        def on_order(self, _i):
            new = self.cb_order.currentData() or "taohe"
            if new == self.order:
                return
            self.order = new
            self.forms = formset_default(self.shape, self.cut, self.order)
            self.build_gallery()
            self.on_pick(next(t for t in self.thumbs if t.form.level == 3))
            self.status("十干排法：" + self.cb_order.currentText())

        def on_cut(self, _i):
            new = self.cb_cut.currentData() or "line"
            if new == self.cut:
                return
            self.cut = new
            self.forms = formset_default(self.shape, self.cut, self.order)
            self.build_gallery()
            self.apply_default_orient()
            self.on_pick(next(t for t in self.thumbs
                              if t.form.level == 3 and t.form.index == 0))
            n = len(form_table(self.cut, 3))
            self.status(f"分割样式：{CUT_STYLES[self.cut]}　·　四分层 {n} 个形态")

        def on_pick(self, thumb):
            for t in self.thumbs:
                t.selected = (t is thumb)
            self.cur = thumb.form
            self.prev.form = self.cur
            nm, ncell, nform = level_info(
                self.cur.level, self.cur.cut if self.shape == "circle" else "line")
            self.cb_order.setEnabled(self.shape == "circle" and self.cut == "taiji")
            self.lb_sel.setText(f"{nm}　第 {self.cur.index+1} / {nform} 个"
                                f"　（{ncell} 格）")
            self._loading = True
            self.ed_name.setText(self.cur.name)
            while self.cell_lay.count():
                w = self.cell_lay.takeAt(0).widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()
            self.cell_cks = []
            labs = self.cur.labels
            for i, lb in enumerate(labs):
                ck = QCheckBox(lb)
                ck.setChecked(i in self.cur.filled)
                ck.toggled.connect(lambda _v, k=i: self.on_cell(k))
                self.cell_lay.addWidget(ck, i // 2, i % 2)
                self.cell_cks.append(ck)
            self._loading = False
            self.refresh()

        def on_cell(self, k):
            if self._loading or self.cur is None:
                return
            s = set(self.cur.filled)
            s.symmetric_difference_update({k})
            self.cur.filled = tuple(sorted(s))
            self.refresh()
            self.status(f"{self.cur.name}：填阳格 = "
                        f"{'、'.join(self.cur.labels[i] for i in self.cur.filled) or '无'}")

        def on_name(self, s):
            if self.cur is not None:
                self.cur.name = s
                self.refresh()

        def on_reset(self):
            if self.cur is None:
                return
            c = self.cut if self.shape == "circle" else "line"
            nm, fl = form_table(c, self.cur.level, self.order)[self.cur.index]
            self.cur.name = nm
            self.cur.filled = tuple(fl)
            self.cur.colors.clear()
            self.on_pick(next(t for t in self.thumbs if t.form is self.cur))

        def pick(self, key):
            c = QColorDialog.getColor(QColor(self.style[key]), self, "选择颜色")
            if c.isValid():
                self.style[key] = c.name()
                self.refresh()

        def on_gap(self, v):
            self.style["hatch_spacing"] = float(v)
            self.refresh()

        def on_show(self, _v):
            self.prev.show_vertices = self.ck_v.isChecked()
            self.prev.show_grid = self.ck_g.isChecked()
            self.prev.update()

        # ---------------------------------------------------- 输出 --
        def _base(self):
            shp = "圆" if self.shape == "circle" else "方"
            cut = "" if self.cut == "line" else "_" + self.cut
            return f"{shp}{cut}_{self.cur.index+1}_{safe(self.cur.name)}"

        def exp_svg(self):
            if self.cur is None:
                return
            p, _ = QFileDialog.getSaveFileName(self, "导出 SVG",
                                               self._base() + ".svg", "SVG (*.svg)")
            if p:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(form_to_svg(self.cur, self.style, 900,
                                        show_vertices=self.ck_v.isChecked(),
                                        show_grid=self.ck_g.isChecked(),
                                        caption_note=True,
                                        rotation=self.rotation))
                self.status("已导出 " + p)

        def exp_png(self):
            if self.cur is None:
                return
            p, _ = QFileDialog.getSaveFileName(self, "导出 PNG",
                                               self._base() + ".png", "PNG (*.png)")
            if p:
                cut0 = self.cut if self.shape == "circle" else "line"
                ops = build_form_ops(self.cur, self.style,
                                     show_vertices=self.ck_v.isChecked(),
                                     show_grid=self.ck_g.isChecked(),
                                     caption_note=True,
                                     rotation=self.rotation)
                render_ops_png(ops, form_bbox(True, True, cut0), p,
                               self.style, 1200)
                self.status("已导出 " + p)

        def exp_sheet(self):
            shp = "圆" if self.shape == "circle" else "方"
            p, _ = QFileDialog.getSaveFileName(self, "导出整张形态表",
                                               f"{shp}_形态表.svg", "SVG (*.svg)")
            if p:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(formset_to_svg(self.forms, self.style,
                                           show_vertices=self.ck_v.isChecked(),
                                           show_grid=self.ck_g.isChecked(),
                                           rotation=self.rotation))
                self.status("已导出 " + p)

        def exp_sheet_png(self):
            shp = "圆" if self.shape == "circle" else "方"
            p, _ = QFileDialog.getSaveFileName(self, "导出整张形态表",
                                               f"{shp}_形态表.png", "PNG (*.png)")
            if not p:
                return
            cut0 = self.cut if self.shape == "circle" else "line"
            bb = form_bbox(True, True, cut0)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            cw, ch = w * 1.12, h * 1.12
            rows = [[f for f in self.forms if f.level == lv] for lv in LEVELS]
            ncol = max(len(r) for r in rows)
            ops = []
            for r, row in enumerate(rows):
                off = (ncol - len(row)) / 2.0
                for c, f in enumerate(row):
                    for op in build_form_ops(
                            f, self.style, caption_note=True,
                            show_vertices=self.ck_v.isChecked(),
                            show_grid=self.ck_g.isChecked(),
                            rotation=self.rotation):
                        ops.append(_shift_op(op, (off + c) * cw - bb[0],
                                             (len(rows) - 1 - r) * ch - bb[1]))
            render_ops_png(ops, (0.0, 0.0, cw * ncol, ch * 3), p, self.style, 2400)
            self.status("已导出 " + p)

        def check_taohe(self):
            if not (self.shape == "circle" and self.cut == "taiji"):
                QMessageBox.information(self, "套合校验",
                                        "只有「圆 · 太极」的十形态才有首尾套合。")
                return
            lines = [f"排法：{self.cb_order.currentText()}", ""]
            ok = True
            for a, ka, b, kb, good, note in taohe_pairs(self.order):
                lines.append(f"  {'✓' if good else '!'} {a}（{ka}） + "
                             f"{b}（{kb}） → {note}")
                ok &= good
            lines += ["",
                      "戊、己 是十形态里仅有的两个「自补」形态："
                      "交错 与 一仪 各自转 180° 就补成全阳，",
                      "所以它们只能自己套自己，跨对补不满 —— 这是几何性质，不是排错。"]
            dlg = QDialog(self); dlg.setWindowTitle("套合校验")
            v = QVBoxLayout(dlg); te = QTextEdit(); te.setReadOnly(True)
            te.setPlainText("\n".join(lines)); te.setMinimumSize(520, 300)
            v.addWidget(te)
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            bb.accepted.connect(dlg.accept); v.addWidget(bb); dlg.exec()

        def save_json(self):
            shp = "圆" if self.shape == "circle" else "方"
            p, _ = QFileDialog.getSaveFileName(self, "保存 JSON",
                                               f"{shp}_形态表.json", "JSON (*.json)")
            if not p:
                return
            d = {"format": "guaxing-form/1", "shape": self.shape,
                 "cut": self.cut, "order": self.order,
                 "rotation": self.rotation,
                 "style": self.style,
                 "forms": [{"level": f.level, "index": f.index, "name": f.name,
                            "filled": list(f.filled),
                            "colors": {str(k): v for k, v in f.colors.items()}}
                           for f in self.forms]}
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False, indent=2)
            self.status("已保存 " + p)

        def load_json(self):
            p, _ = QFileDialog.getOpenFileName(self, "读取 JSON", "", "JSON (*.json)")
            if not p:
                return
            try:
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
                if d.get("shape") != self.shape:
                    raise ValueError("这份存档是另一个形状的")
                self.style.update(d.get("style", {}))
                self.rotation = int(d.get("rotation", 0)) % 360
                self.cut = d.get("cut", "line")
                self.order = d.get("order", "taohe")
                self.forms = formset_default(self.shape, self.cut, self.order)
                for rec in d.get("forms", []):
                    for f in self.forms:
                        if f.level == rec["level"] and f.index == rec["index"]:
                            f.name = rec.get("name", f.name)
                            f.filled = tuple(rec.get("filled", f.filled))
                            f.colors = {int(k): v for k, v
                                        in rec.get("colors", {}).items()}
            except Exception as e:
                QMessageBox.warning(self, "读取失败", str(e)); return
            self._loading = True
            self.sp_gap.setValue(self.style.get("hatch_spacing", 0.075))
            self.cb_cut.setCurrentIndex(max(0, self.cb_cut.findData(self.cut)))
            self.cb_order.setCurrentIndex(max(0, self.cb_order.findData(self.order)))
            self.lb_rot.setText(f"{self.rotation}°")
            self._loading = False
            self.build_gallery()
            self.on_pick(self.thumbs[0])
            self.status("已读取 " + p)

    # ---------------------------------------------------------- 主窗 --
    class Main(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("卦形几何生成器  guaxing")
            self.resize(1320, 880)
            tabs = QTabWidget()
            st = self.statusBar().showMessage
            self.tri = TriangleTab(st)
            tabs.addTab(self.tri, "三角形（2ⁿ 卦形）")
            tabs.addTab(FormTab("circle", st), "圆（形态表）")
            tabs.addTab(FormTab("square", st), "方（形态表）")
            self.setCentralWidget(tabs)
            self.statusBar().showMessage("就绪")

    app = QApplication(sys.argv)
    app.setApplicationName("guaxing")
    w = Main(); w.show()
    return app.exec()


# =============================================================================
# 第 6 部分：命令行
# =============================================================================

def selftest(last_cut="median"):
    print("=" * 66)
    print(f"guaxing 几何自检   末两爻切法 = "
          f"{'中线' if last_cut == 'median' else '斜切'}")
    print("=" * 66)
    all_ok = True
    for n in GUA_LEVELS:
        for mode in ("gua", "even"):
            lay = build_layout(n, mode, last_cut=last_cut)
            ok, msgs = validate_layout(lay)
            all_ok &= ok
            tag = "卦形" if mode == "gua" else "均分"
            print(f"\n--- n={n}  {tag}  ({2**n} 组合 / {lay.n_cells} 格) ---")
            for s in msgs:
                print("   " + s)

    # 与 GeoGebra 原图逐卦比对：八卦的填色格集合
    print("\n" + "=" * 66)
    print("与原 GeoGebra 图比对（八卦，n=3）")
    print("=" * 66)
    lay = build_layout(3)
    labels = {}
    for li in range(3):
        for ci in lay.lines[li]:
            labels[ci] = lay.yao_label(li)
    # 依几何位置命名格子
    def where(t):
        c = t.centroid
        if c[1] > -0.29:
            return "左上" if c[0] < -0.05 else ("右上" if c[0] > 0.05 else "中央")
        return "底部"
    expect = {
        "乾": {"左上", "右上", "中央", "底部"},
        "坤": set(),
        "坎": {"中央"},
        "离": {"左上", "右上", "底部"},
        "震": {"底部"},
        "巽": {"左上", "右上", "中央"},
        "艮": {"左上", "右上"},
        "兑": {"中央", "底部"},
    }
    for code in range(8):
        name = TRIGRAM[code]
        b = bits_of(code, 3)
        filled = set()
        for li in range(3):
            if b[li]:
                for ci in lay.lines[li]:
                    filled.add(where(lay.cells[ci]))
        good = filled == expect[name]
        all_ok &= good
        print(f"  {'✓' if good else '✗'} {name} {gua_symbol(code,3)} "
              f"{code_string(code,3)}  填色格 = "
              f"{'、'.join(sorted(filled)) or '（全空）'}")

    print("\n" + ("全部通过 ✓" if all_ok else "有失败项 ✗"))
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser(description="卦形几何生成器 guaxing")
    ap.add_argument("--selftest", action="store_true", help="纯几何自检（不需要 PyQt6）")
    ap.add_argument("--batch", metavar="OUTDIR", help="批量导出到目录")
    ap.add_argument("--n", type=int, default=6, choices=GUA_LEVELS,
                    help="层数 n，只有 1 / 2 / 3 / 6（无四爻五爻）；默认 6")
    ap.add_argument("--fmt", default="svg", choices=["svg"], help="命令行只支持 svg")
    ap.add_argument("--scheme", default="binary",
                    choices=["binary", "xiantian", "lianshan"])
    ap.add_argument("--px", type=int, default=520)
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--mirror", action="store_true", help="左右镜像")
    ap.add_argument("--low-first", action="store_true", help="大块归初爻")
    ap.add_argument("--point-up", action="store_true", help="尖朝上")
    ap.add_argument("--cut", default="median", choices=["median", "oblique"],
                    help="末两爻切法：median 中线（默认）/ oblique 斜切")
    a = ap.parse_args()

    if a.selftest:
        return selftest(a.cut)
    if a.batch:
        files = batch_export(a.batch, n=a.n, mirror=a.mirror,
                             high_first=not a.low_first,
                             point_down=not a.point_up, last_cut=a.cut,
                             scheme=a.scheme, fmt="svg", px=a.px,
                             sheet=not a.no_sheet)
        print(f"写出 {len(files)} 个文件到 {a.batch}")
        return 0
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
