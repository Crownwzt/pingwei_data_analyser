"""
筛选测试集"高价值待解决"的预测误差场景。

不是简单按数值大小排，而是按"业务影响 × 可改进空间"筛出 5 类典型问题：

  Case A: 谷底价漏报（真实值 < 50 而预测 > 150） - MAPE 爆表主因
  Case B: 尖峰价漏报（真实值 > 500 且预测低估 > 100） - 交易风险最大
  Case C: 方向误判（真实↑ 但预测↓ 或反之，且幅度 > 50）
  Case D: 陡变时段（相邻小时真实值变化 > 100 但预测平滑）
  Case E: 大偏差持续（连续 3h 以上 |err| > 50）

每类挑 Top N 案例，画对应"上小时+下15min"双子图。

产物：
  critical_cases/
    ├── A_valley_miss/           谷底漏报 Top N
    ├── B_peak_miss/             尖峰漏报 Top N
    ├── C_direction_flip/        方向误判 Top N
    ├── D_ramp_smooth/           陡变平滑 Top N
    ├── E_persistent_bias/       持续偏差 Top N
    └── overview.html            总览页
"""
from __future__ import annotations

import base64
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
METRICS_PKL = os.path.join(PROJECT, "outputs", "metrics.pkl")
XLSX_15MIN = os.path.join(
    PROJECT, "2025-2026市场情况", "2025-12-01到2025-12-30市场价格趋势.xlsx",
)
CASES_DIR = os.path.join(ROOT, "critical_cases")
os.makedirs(CASES_DIR, exist_ok=True)

TOP_N_PER_CASE = 5


def setup_font():
    cand = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in cand if c in avail), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(METRICS_PKL, "rb") as f:
        m = pickle.load(f)
    hourly = pd.DataFrame({
        "datetime": pd.to_datetime(m["dt_test"]),
        "y_true": m["y_test"],
        "da": m["da_test"],
        "y_pred": m["metrics"]["test"]["y_pred"],
    })
    hourly["date"] = hourly["datetime"].dt.date
    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["abs_err"] = (hourly["y_true"] - hourly["y_pred"]).abs()
    hourly["signed_err"] = hourly["y_true"] - hourly["y_pred"]

    xl = pd.ExcelFile(XLSX_15MIN)
    d = xl.parse("明细")
    d["datetime"] = pd.to_datetime(d["日期"].astype(str) + " " + d["时间"].astype(str))
    d = d.rename(columns={
        "日前统一结算点电价(元/MWh)": "da",
        "实时统一结算点电价(元/MWh)": "y_true",
    })
    d["date"] = d["datetime"].dt.date
    f15 = d[["datetime", "date", "y_true", "da"]].sort_values("datetime").reset_index(drop=True)
    return hourly, f15


# ---------- 案例筛选逻辑 ----------
def case_A_valley_miss(hourly: pd.DataFrame) -> pd.DataFrame:
    """真实值是谷底（< 80）但模型预测正常水位（> 150）"""
    mask = (hourly["y_true"] < 80) & (hourly["y_pred"] > 150)
    return hourly[mask].nlargest(TOP_N_PER_CASE, "abs_err")


def case_B_peak_miss(hourly: pd.DataFrame) -> pd.DataFrame:
    """真实值高价（> 500）且模型显著低估（低估 > 80）"""
    mask = (hourly["y_true"] > 500) & (hourly["signed_err"] > 80)
    return hourly[mask].nlargest(TOP_N_PER_CASE, "signed_err")


def case_C_direction_flip(hourly: pd.DataFrame) -> pd.DataFrame:
    """相邻小时真实值变化方向与预测变化方向相反，且真实幅度 > 50"""
    df = hourly.sort_values("datetime").reset_index(drop=True).copy()
    df["dy_true"] = df["y_true"].diff()
    df["dy_pred"] = df["y_pred"].diff()
    # 同一天内比较
    df["same_day"] = df["date"] == df["date"].shift(1)
    mask = (
        df["same_day"]
        & (df["dy_true"].abs() > 50)
        & (np.sign(df["dy_true"]) != np.sign(df["dy_pred"]))
    )
    df["flip_score"] = df["dy_true"].abs() + df["dy_pred"].abs()
    return df[mask].nlargest(TOP_N_PER_CASE, "flip_score")


def case_D_ramp_smooth(hourly: pd.DataFrame) -> pd.DataFrame:
    """真实值陡变（相邻 h |Δ| > 100）但预测变化 < 30"""
    df = hourly.sort_values("datetime").reset_index(drop=True).copy()
    df["dy_true"] = df["y_true"].diff()
    df["dy_pred"] = df["y_pred"].diff()
    df["same_day"] = df["date"] == df["date"].shift(1)
    mask = (
        df["same_day"]
        & (df["dy_true"].abs() > 100)
        & (df["dy_pred"].abs() < 30)
    )
    df["smooth_score"] = df["dy_true"].abs() - df["dy_pred"].abs()
    return df[mask].nlargest(TOP_N_PER_CASE, "smooth_score")


def case_E_persistent_bias(hourly: pd.DataFrame) -> pd.DataFrame:
    """连续 3h 以上 |err| > 50，并返回每段的起点行"""
    df = hourly.sort_values("datetime").reset_index(drop=True).copy()
    df["is_bad"] = df["abs_err"] > 50
    # 找连续段
    df["seg_id"] = (df["is_bad"] != df["is_bad"].shift(1)).cumsum()
    segs = []
    for sid, g in df[df["is_bad"]].groupby("seg_id"):
        if len(g) >= 3:
            # 段内的日期必须一致（不跨天）
            if g["date"].nunique() == 1:
                segs.append({
                    "date": g["date"].iloc[0],
                    "start_hour": int(g["hour"].iloc[0]),
                    "end_hour": int(g["hour"].iloc[-1]),
                    "n_hours": len(g),
                    "mean_abs_err": float(g["abs_err"].mean()),
                    "max_abs_err": float(g["abs_err"].max()),
                    "datetime": g["datetime"].iloc[0],
                })
    return (pd.DataFrame(segs)
            .sort_values("mean_abs_err", ascending=False)
            .head(TOP_N_PER_CASE)
            if segs else pd.DataFrame())


# ---------- 绘图 ----------
def plot_case(hourly_day: pd.DataFrame, fifteen_day: pd.DataFrame,
              highlight_hours: List[int], title: str, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )
    hd = hourly_day.sort_values("datetime")

    # 沿用 outputs/daily/test 的风格
    ax1.plot(hd["datetime"], hd["y_true"], label="真实",
             color="#4C72B0", lw=1.8, marker="o", ms=4)
    ax1.plot(hd["datetime"], hd["da"], label="日前价(B7')",
             color="#888", lw=1.2, ls=":", marker="s", ms=3)
    ax1.plot(hd["datetime"], hd["y_pred"], label="XGB 预测",
             color="#C44E52", lw=1.6, marker="x", ms=5)
    ax1.fill_between(hd["datetime"], hd["y_true"], hd["y_pred"],
                     color="#C44E52", alpha=0.12, label="XGB 误差区间")

    # 高亮问题小时段
    day_ts = hd["datetime"].dt.floor("D").iloc[0]
    for h in highlight_hours:
        left = day_ts + pd.Timedelta(hours=h - 0.5)
        right = day_ts + pd.Timedelta(hours=h + 0.5)
        ax1.axvspan(left, right, color="red", alpha=0.12)

    ax1.set_ylabel("电价 (元/MWh)")
    ax1.set_title(title, fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="best")

    if len(fifteen_day) > 0:
        fd = fifteen_day.sort_values("datetime")
        ax2.plot(fd["datetime"], fd["y_true"], label="实时 15min",
                 color="#4C72B0", lw=1.2)
        ax2.plot(fd["datetime"], fd["da"], label="日前 15min",
                 color="#888", lw=1.2, ls=":")
        ax2.fill_between(fd["datetime"], fd["y_true"], fd["da"],
                         where=fd["y_true"] > fd["da"],
                         color="#C44E52", alpha=0.12, label="实时>日前")
        ax2.fill_between(fd["datetime"], fd["y_true"], fd["da"],
                         where=fd["y_true"] < fd["da"],
                         color="#4C72B0", alpha=0.10, label="实时<日前")
        for h in highlight_hours:
            left = day_ts + pd.Timedelta(hours=h)
            right = day_ts + pd.Timedelta(hours=h + 1)
            ax2.axvspan(left, right, color="red", alpha=0.10)

    ax2.set_xlabel("时间")
    ax2.set_ylabel("15min 电价 (元/MWh)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


# ---------- 主流程 ----------
def render_overview_html(case_summary: Dict[str, pd.DataFrame],
                        case_meta: Dict[str, Dict], out_path: str) -> None:
    """生成总览 HTML，列出各类案例的图片和说明"""
    def img_b64(p: str) -> str:
        if not os.path.exists(p):
            return ""
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()

    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>测试集高价值误差场景</title>
<style>
body { font-family: -apple-system, "PingFang SC", sans-serif;
       max-width: 1200px; margin: 20px auto; padding: 0 16px; }
h1 { border-bottom: 3px solid #d9534f; padding-bottom: 8px; }
h2 { color: #d9534f; margin-top: 30px; border-left: 5px solid #d9534f; padding-left: 10px; }
.case { border: 1px solid #ddd; padding: 12px; margin: 12px 0; border-radius: 5px; background: #fafafa; }
.desc { color: #555; margin: 8px 0; font-size: 14px; }
img { max-width: 100%; border: 1px solid #eee; }
table { border-collapse: collapse; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 5px 8px; }
th { background: #f0f0f0; }
.hint { background: #fff4e5; border-left: 4px solid #f0ad4e;
        padding: 10px 14px; margin: 10px 0; }
</style></head><body>
<h1>测试集高价值预测误差场景（诊断集）</h1>
<div class='hint'>
<p>本页按<strong>业务影响</strong>而非纯数值大小筛选出 5 类典型问题，每类挑 Top """ + str(TOP_N_PER_CASE) + """ 案例，
辅助定位模型的失效模式，指导下一步特征/数据获取方向。</p>
</div>
"""]
    for case_key, meta in case_meta.items():
        cases = case_summary.get(case_key, pd.DataFrame())
        parts.append(f"<h2>{meta['title']}</h2>")
        parts.append(f"<div class='desc'>{meta['desc']}</div>")
        parts.append(f"<div class='hint'><strong>业务含义</strong>：{meta['business']}<br>"
                     f"<strong>可能成因</strong>：{meta['cause']}<br>"
                     f"<strong>改进方向</strong>：{meta['fix']}</div>")
        if cases.empty:
            parts.append("<p><em>本类别无匹配样本</em></p>")
            continue

        subdir = os.path.join(CASES_DIR, meta["dir"])
        for _, row in cases.iterrows():
            date = row.get("date", None)
            # 图片文件名
            files = [f for f in os.listdir(subdir) if str(date) in f] if os.path.exists(subdir) else []
            if not files:
                continue
            img_path = os.path.join(subdir, sorted(files)[0])
            b64 = img_b64(img_path)
            if b64:
                parts.append(f"<div class='case'><h4>{date}  h={row.get('hour', row.get('start_hour', '?'))}</h4>")
                parts.append(f'<img src="data:image/png;base64,{b64}"></div>')

    # ---------- 10 个关键问题（面向数据提供方）----------
    parts.append("""
<h2 style="border-left-color:#337ab7;color:#337ab7;">附：10 个关键问题（面向数据提供方）</h2>
<div class='hint' style='border-left-color:#337ab7;background:#eef5fb;'>
以下 10 个问题直接对应上方 5 类无法用现有特征解释的预测现象。
带着这些问题去问数据提供方（省调 / 交易中心 / 电厂运行部），
每一条都指向一个具体的<strong>数据颗粒度</strong>或<strong>业务口径</strong>诉求。
</div>
<ol style="line-height:1.8;font-size:14px;">

<li><strong>[对应 A 谷底漏报]</strong> 12-26 / 12-29 / 12-25 的午间时段出现了 &lt; 30 元/MWh 甚至接近 0 的实时价，
<em>请问这些时段是否触发了 <strong>"新能源消纳受限 / 边界机组报价触底 / 阻塞导致本地价格塌陷"</strong>
中的哪一类机制？我们已有 96 点新能源日前预报与实际出力，但缺少<strong>系统调峰能力剩余量</strong>
和<strong>是否触发新能源限电</strong>的指示信号，这类调度指令能否按日/时段提供？</em></li>

<li><strong>[对应 A]</strong> 12-26 午间光伏日前预报值远大于实际（可以推出正向偏差），
但日前价并没预见到消纳压力（日前 &gt; 100 而实时 &lt; 30）。
<em>请问日前出清时使用的<strong>新能源预报是否与"供需情况.xlsx > 日前 sheet"中的新能源负荷字段一致</strong>？
如果一致，为什么日前市场没据此下修价格？是否存在<strong>日内滚动更新的新能源预报</strong>
（即 D 日当天每 2-4h 反复更新）？如有，能否提供？</em></li>

<li><strong>[对应 B 尖峰漏报]</strong> 12-XX 出现 &gt; 500 元/MWh 的尖峰，我们模型和日前都低估。
<em>请问该时段是否发生了<strong>机组非计划停运 (UFOR) / 联络线临时降容 / 备用不足触发高价段</strong>？
能否提供该日的<strong>实时可用装机容量</strong>与<strong>备用充裕度</strong>（此二者当前数据中没有）？</em></li>

<li><strong>[对应 C 方向误判]</strong> 我们观测到相邻小时电价方向被完全反转的案例（如真 +80 元但预 −80 元）。
数据里已有"外来（送）负荷(MW)"的<em>日前</em>值，但没有实际值。
<em>请问外来送电的<strong>实际曲线（实时执行的联络线功率）</strong>能否补齐 15min 粒度？
这块偏差可能是"日前价平稳但实时价急拐弯"的关键成因之一。</em></li>

<li><strong>[对应 C]</strong> 谷 → 平 / 平 → 峰 的过渡时段（早 6-8h、晚 21-22h）方向反转集中出现。
<em>请问该时段的<strong>爬坡辅助服务出清价</strong>与<strong>调用容量</strong>是否可获得？
这些量应该能解释一部分过渡时段的价格反转。</em></li>

<li><strong>[对应 D 陡变平滑]</strong> 我们看到多个案例中，真实价 1 小时内变化 &gt; 100 元/MWh，
但基础面（负荷、新能源、光伏、风电）实际值只有小幅变化。
<em>数据现有 <code>实时节点电价</code>与<code>实时统一结算点电价</code>二者的差值（10-30 元/MWh 波动），
请问：该差值是否就是<strong>阻塞成本 + 网损分摊</strong>？如果是，我们可用其反推阻塞事件时段；
如不是，请说明其形成机制。同时能否提供<strong>阻塞时段与阻塞节点</strong>的官方记录？</em></li>

<li><strong>[对应 D]</strong> 我们已有 15min 粒度的光伏实际出力和光伏日前预报（可算偏差）。
<em>但缺少<strong>光伏集合预报的分歧度</strong>（不同气象源或不同时间戳预报的方差），
这个不确定性代理量能否作为独立数据源提供？此外，风电爬坡时段的<strong>短临风功率预报（0-4h）</strong>
是否可获取？我们目前只有日前预报。</em></li>

<li><strong>[对应 E 持续偏差]</strong> 12-05、12-20 等日出现整段（连续 3-6 小时）系统性偏差。
<em>请问这些日子是否发生了<strong>大机组集中检修、极端天气事件、跨省送电计划大幅调整</strong>？
能否按日提供当日<strong>省调运行方式说明</strong>或调度事件记录？</em></li>

<li><strong>[对应 E 结构性]</strong> 我们发现所有 Δ（实际−日前）物理量组合起来只能解释 8.8% 的实时价偏差方差，
其余 91% 在现有数据外。<em>请问是否存在我们未接入的<strong>结构性数据</strong>：
（a）机组报价曲线；（b）阻塞时段与阻塞节点；（c）备用调用记录；（d）联络线实时功率？
这四类数据分别属于哪个部门归口，能否申请脱敏后的历史样本？</em></li>

<li><strong>[口径确认 / 反推信号]</strong> 我们的预测目标是<code>实时统一结算点电价</code>（全省结算基准价），
但数据里同时有<code>实时节点电价</code>（本地节点 LMP），二者差值波动 10-30 元/MWh。
<em>请澄清二者的<strong>形成机制</strong>：<strong>Δ = 节点价 − 统一价 = 阻塞成本 + 网损分摊</strong>是否成立？
若成立，我们可用该差值的历史序列反推阻塞频次与节点效应，替代直接获取阻塞记录；
若不成立，请说明真实构成。补充：<code>实时节点电价</code>与目标同时刻产生，已被识别为泄漏并排除，
本问不是要将其作为预测特征，而是<strong>作为诊断信号使用</strong>。</em></li>

</ol>

<div class='hint' style='border-left-color:#5cb85c;background:#eef7ee;'>
<strong>使用建议</strong>：把上述问题带到与数据方的下一次沟通中，按"是否可获取 / 颗粒度 / 历史长度 / 更新频率"四个维度记录反馈。
优先落实第 3、6、9、10 条 —— 它们分别对应<em>尖峰、陡变（阻塞诊断）、结构性偏差、目标口径确认</em>四个最大的
误差来源，也是投入产出比最高的数据获取方向。
</div>
<div class='hint' style='border-left-color:#5cb85c;background:#eef7ee;'>
<strong>已确认数据中<u>已有</u>的项（无需再问）</strong>：
15min 粒度的日前/实际负荷、新能源、光伏、风电、水电、非市场化出力、竞价空间；
15min 粒度的日前/实时统一结算点电价、日前/实时节点电价；负荷率、外来送电<em>日前</em>值。
</div>
""")
    parts.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    setup_font()
    print("[1/3] 加载数据")
    hourly, f15 = load_data()

    print("[2/3] 筛选 5 类案例")
    case_A = case_A_valley_miss(hourly)
    case_B = case_B_peak_miss(hourly)
    case_C = case_C_direction_flip(hourly)
    case_D = case_D_ramp_smooth(hourly)
    case_E = case_E_persistent_bias(hourly)

    case_meta = {
        "A": {
            "title": "Case A: 谷底价漏报（真<80 但预>150）",
            "dir": "A_valley_miss",
            "desc": "真实价跌入深谷（负荷率极低/新能源大发），但模型输出正常水位。",
            "business": "买方在低价时段仍按高价采购，直接经济损失；也是 MAPE 爆表的主因。",
            "cause": "新能源大发+负荷谷底叠加时，模型见过的样本少；日前预报也未及时下修。",
            "fix": "加入 新能源占比的滚动波动性、当日光伏预报误差、日间负荷极小值 等特征。",
        },
        "B": {
            "title": "Case B: 尖峰价漏报（真>500 且低估>80）",
            "dir": "B_peak_miss",
            "desc": "真实价冲高但模型平滑掉了尖峰。",
            "business": "卖方在高价时段错失套利；风险管理关键指标。",
            "cause": "机组非计划停运、备用不足、报价策略性拉抬 —— 均在模型可见特征之外。",
            "fix": "获取实时可用容量、备用率、机组检修计划；改用分位数损失突出上尾。",
        },
        "C": {
            "title": "Case C: 方向误判（相邻 h 真↑预↓ 或反）",
            "dir": "C_direction_flip",
            "desc": "模型判断的涨跌方向与实际相反。",
            "business": "直接损害交易信号可用性，即使幅度接近也会误导决策。",
            "cause": "自回归依赖偏强或时段过渡（谷→平/平→峰）特征不足。",
            "fix": "增加 时段跃变哑变量、上/下爬坡指示、前一小时误差反馈。",
        },
        "D": {
            "title": "Case D: 陡变时段被平滑（真|Δ|>100 但预|Δ|<30）",
            "dir": "D_ramp_smooth",
            "desc": "真实价相邻小时快速变化，模型输出趋于平坦。",
            "business": "无法及时响应爬坡时段（早晚高峰起爬/退爬），影响短时策略。",
            "cause": "特征工程未刻画爬坡强度；XGB 对时序位置感弱。",
            "fix": "增加 (h-1) 到 (h+1) 的负荷/新能源变化速率作为特征。",
        },
        "E": {
            "title": "Case E: 大偏差持续（连续 3h+ |err|>50）",
            "dir": "E_persistent_bias",
            "desc": "误差不是孤立点，而是整段时间跑偏。",
            "business": "整段决策全错，累计损失最大；提示某类结构性偏置。",
            "cause": "该日整体供需模式偏离训练集分布；系统性事件未被特征捕获。",
            "fix": "残差自回归、当日模型偏置修正、加入日级别的宏观特征（煤价、天气）。",
        },
    }
    case_summary = {"A": case_A, "B": case_B, "C": case_C, "D": case_D, "E": case_E}

    print("[3/3] 生成案例图")
    for key, cases in case_summary.items():
        meta = case_meta[key]
        subdir = os.path.join(CASES_DIR, meta["dir"])
        os.makedirs(subdir, exist_ok=True)
        if cases.empty:
            print(f"      {key}: 0 案例")
            continue

        for i, row in cases.iterrows():
            date = row["date"]
            hourly_day = hourly[hourly["date"] == date]
            f15_day = f15[f15["date"] == date]

            if "hour" in row.index and not pd.isna(row.get("hour", np.nan)):
                highlight = [int(row["hour"])]
                hour_tag = f"h{int(row['hour']):02d}"
            elif "start_hour" in row.index:
                highlight = list(range(int(row["start_hour"]), int(row["end_hour"]) + 1))
                hour_tag = f"h{int(row['start_hour']):02d}-{int(row['end_hour']):02d}"
            else:
                highlight = []
                hour_tag = "day"

            # 标题信息
            if key == "A":
                subtitle = f"真={row['y_true']:.0f}  预={row['y_pred']:.0f}  err={row['abs_err']:.0f}"
            elif key == "B":
                subtitle = f"真={row['y_true']:.0f}  预={row['y_pred']:.0f}  低估={row['signed_err']:.0f}"
            elif key == "C":
                subtitle = f"Δ真={row['dy_true']:+.0f}  Δ预={row['dy_pred']:+.0f}"
            elif key == "D":
                subtitle = f"Δ真={row['dy_true']:+.0f}  Δ预={row['dy_pred']:+.0f}"
            else:
                subtitle = f"{row['start_hour']}-{row['end_hour']}h  平均err={row['mean_abs_err']:.0f}  持续 {row['n_hours']}h"

            title = f"[{meta['title']}] {date} {hour_tag}  |  {subtitle}"
            fname = f"{date}_{hour_tag}.png"
            plot_case(hourly_day, f15_day, highlight, title,
                      os.path.join(subdir, fname))

        print(f"      {key}: {len(cases)} 案例 → {subdir}")

    print("      生成总览 HTML")
    render_overview_html(case_summary, case_meta,
                         os.path.join(CASES_DIR, "overview.html"))
    print(f"\n完成。总览: {os.path.join(CASES_DIR, 'overview.html')}")


if __name__ == "__main__":
    main()
