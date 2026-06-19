import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Polygon


# -----------------------------
# Model inputs
# -----------------------------

N_SIMS = 10_000
N_YEARS = 5
YEARS = np.array([2025, 2026, 2027, 2028, 2029])

ML_PARAMS = {
    "base_rev_growth": 0.1736,
    "rev_growth_std": 0.0583,
    "base_op_margin": 0.142075,
    "op_to_net_coef": 0.5759,
    "initial_revenue": 11_313.853,  # $M, 2024 actual
    "initial_op_income": 1_916.332,  # $M, 2024 actual
    "initial_fcf": 1_511.473,  # $M, 2024 actual
}

SOLUTIONS = {
    "subscription": {
        "name": "Chips & Drinks Subscription",
        "rev_boost": (0.015, 0.030, 0.050),
        "margin_boost": (0.005, 0.010, 0.018),
        "weight": 0.35,
    },
    "float": {
        "name": "Stored-Value Float Strategy",
        "rev_boost": (0.005, 0.012, 0.020),
        "margin_boost": (0.008, 0.015, 0.025),
        "weight": 0.25,
    },
    "international": {
        "name": "International Expansion",
        "rev_boost": (0.008, 0.020, 0.035),
        "margin_boost": (0.004, 0.010, 0.018),
        "weight": 0.40,
    },
}

SCENARIOS = {
    "bear": {"macro_drag": -0.025, "strategy_mult": 0.40, "label": "Bear"},
    "base": {"macro_drag": 0.000, "strategy_mult": 1.00, "label": "Base"},
    "bull": {"macro_drag": 0.010, "strategy_mult": 1.60, "label": "Bull"},
}

DCF_PARAMS = {
    "wacc": 0.0855,
    "terminal_growth": 0.030,
    "shares_out_b": 1.290,
    "cash_b": 1.200,
    "debt_b": 0.000,
}

# OLS coefficients used by the notebook model.
INTERCEPT_A = -1.3892
BETA1_A = 31.8584
BETA2_A = 0.6927


# -----------------------------
# Styling
# -----------------------------

BG = "#F7F1E8"
RED = "#C94235"
BROWN = "#4A1B0D"
GOLD = "#D9AE3F"
GRID = "#E8DDCF"
EDGE = "#6A3A2C"
RED_LIGHT = "#F17868"
GOLD_LIGHT = "#F7D572"
BASELINE = "#7A3F28"
BASELINE_LIGHT = "#B9784D"
BASELINE_FILL = "#A86642"
PANEL = "#F0E7DA"
GREEN = "#3F6F37"
ORANGE = "#D86A32"
FONT_FAMILY = [
    "Avenir Next",
    "Avenir",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "sans-serif",
]

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": BROWN,
        "axes.labelcolor": BROWN,
        "axes.titlecolor": BROWN,
        "text.color": BROWN,
        "xtick.color": BROWN,
        "ytick.color": BROWN,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "grid.color": GRID,
        "grid.alpha": 0.38,
        "grid.linewidth": 0.65,
        "font.size": 11,
        "font.weight": "regular",
        "legend.facecolor": BG,
        "legend.edgecolor": GRID,
        "legend.framealpha": 0.95,
    }
)


# -----------------------------
# Simulation and valuation
# -----------------------------

def compute_boosts(scenario_mult):
    rev_min = sum(s["rev_boost"][0] for s in SOLUTIONS.values()) * scenario_mult
    rev_mode = sum(s["rev_boost"][1] for s in SOLUTIONS.values()) * scenario_mult
    rev_max = sum(s["rev_boost"][2] for s in SOLUTIONS.values()) * scenario_mult
    mar_min = sum(s["margin_boost"][0] for s in SOLUTIONS.values()) * scenario_mult
    mar_mode = sum(s["margin_boost"][1] for s in SOLUTIONS.values()) * scenario_mult
    mar_max = sum(s["margin_boost"][2] for s in SOLUTIONS.values()) * scenario_mult
    return rev_min, rev_mode, rev_max, mar_min, mar_mode, mar_max


def run_strategy(scenario_key="base", seed=42):
    rng = np.random.default_rng(seed)
    scenario = SCENARIOS[scenario_key]
    p = ML_PARAMS
    rev_min, rev_mode, rev_max, mar_min, mar_mode, mar_max = compute_boosts(
        scenario["strategy_mult"]
    )

    revenues = np.zeros((N_SIMS, N_YEARS))
    op_incomes = np.zeros((N_SIMS, N_YEARS))
    net_incomes = np.zeros((N_SIMS, N_YEARS))
    fcfs = np.zeros((N_SIMS, N_YEARS))

    for sim in range(N_SIMS):
        revenue = p["initial_revenue"]
        prev_margin = p["base_op_margin"] * 100
        rev_boost = rng.triangular(rev_min, rev_mode, rev_max)
        margin_boost = rng.triangular(mar_min, mar_mode, mar_max)

        for year_idx in range(N_YEARS):
            base_growth = rng.normal(
                p["base_rev_growth"] + scenario["macro_drag"], p["rev_growth_std"]
            )
            total_growth = base_growth + rev_boost * (0.88**year_idx)
            revenue *= 1 + total_growth

            margin_pct = (
                INTERCEPT_A
                + BETA1_A * total_growth
                + BETA2_A * prev_margin
                + rng.normal(0, 1.5)
                + margin_boost * (0.92**year_idx) * 100
            )
            margin_pct = np.clip(margin_pct, 5.0, 30.0)
            prev_margin = margin_pct

            op_income = revenue * margin_pct / 100
            net_income = op_income * p["op_to_net_coef"] * rng.normal(1.0, 0.03)
            fcf = net_income * rng.normal(0.88, 0.05)

            revenues[sim, year_idx] = revenue
            op_incomes[sim, year_idx] = op_income
            net_incomes[sim, year_idx] = net_income
            fcfs[sim, year_idx] = fcf

    return revenues, op_incomes, net_incomes, fcfs


def run_baseline(seed=99):
    rng = np.random.default_rng(seed)
    p = ML_PARAMS
    revenues = np.zeros((N_SIMS, N_YEARS))
    op_incomes = np.zeros((N_SIMS, N_YEARS))
    net_incomes = np.zeros((N_SIMS, N_YEARS))
    fcfs = np.zeros((N_SIMS, N_YEARS))

    for sim in range(N_SIMS):
        revenue = p["initial_revenue"]
        prev_margin = p["base_op_margin"] * 100

        for year_idx in range(N_YEARS):
            growth = rng.normal(p["base_rev_growth"], p["rev_growth_std"])
            margin_pct = (
                INTERCEPT_A
                + BETA1_A * growth
                + BETA2_A * prev_margin
                + rng.normal(0, 1.5)
            )
            margin_pct = np.clip(margin_pct, 5.0, 25.0)
            revenue *= 1 + growth
            prev_margin = margin_pct

            op_income = revenue * margin_pct / 100
            net_income = op_income * p["op_to_net_coef"] * rng.normal(1.0, 0.03)
            fcf = net_income * rng.normal(0.88, 0.05)

            revenues[sim, year_idx] = revenue
            op_incomes[sim, year_idx] = op_income
            net_incomes[sim, year_idx] = net_income
            fcfs[sim, year_idx] = fcf

    return revenues, op_incomes, net_incomes, fcfs


def dcf_enterprise_value(fcf_array):
    """Return enterprise value, equity value, and price per share arrays."""
    wacc = DCF_PARAMS["wacc"]
    terminal_growth = DCF_PARAMS["terminal_growth"]
    discount_factors = np.array([1 / (1 + wacc) ** t for t in range(1, N_YEARS + 1)])

    pv_fcfs = (fcf_array * discount_factors).sum(axis=1)
    terminal_value = fcf_array[:, -1] * (1 + terminal_growth) / (
        wacc - terminal_growth
    )
    pv_terminal = terminal_value / (1 + wacc) ** N_YEARS

    enterprise_value_m = pv_fcfs + pv_terminal
    equity_value_m = enterprise_value_m + DCF_PARAMS["cash_b"] * 1000 - DCF_PARAMS[
        "debt_b"
    ] * 1000
    price_per_share = (equity_value_m / 1000) / DCF_PARAMS["shares_out_b"]
    return enterprise_value_m / 1000, equity_value_m / 1000, price_per_share


def fixed_baseline_fcf():
    """Notebook valuation convention: baseline FCF grows 15% from 2024 actual."""
    path = np.array([1.511 * (1.15**year) for year in range(1, N_YEARS + 1)]) * 1000
    return np.tile(path, (N_SIMS, 1))


# -----------------------------
# Reporting helpers
# -----------------------------

def p50(array):
    return np.median(array)


def b(value_m):
    return value_m / 1000


def pct_change(new, old):
    return (new / old - 1) * 100


def setup_axis(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, alpha=0.36, linewidth=0.65)
    ax.tick_params(length=0, pad=6)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    return ax


def draw_3d_barh(ax, y, width, height, color, side_color, top_color, edge_color):
    dx = max(width * 0.03, 1.0)
    dy = height * 0.17
    ax.barh(
        y,
        width,
        height=height,
        color=color,
        edgecolor=edge_color,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(
        Polygon(
            [
                (0, y + height / 2),
                (dx, y + height / 2 + dy),
                (width + dx, y + height / 2 + dy),
                (width, y + height / 2),
            ],
            closed=True,
            facecolor=top_color,
            edgecolor=edge_color,
            linewidth=0.8,
            zorder=4,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (width, y - height / 2),
                (width + dx, y - height / 2 + dy),
                (width + dx, y + height / 2 + dy),
                (width, y + height / 2),
            ],
            closed=True,
            facecolor=side_color,
            edgecolor=edge_color,
            linewidth=0.8,
            zorder=4,
        )
    )


def glow_line(ax, x, y, color, label=None, zorder=3, markers=True):
    for linewidth, alpha in [(12, 0.04), (7, 0.08), (3.5, 0.18)]:
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder - 1)
    marker_args = {}
    if markers:
        marker_args = {
            "marker": "o",
            "markersize": 5,
            "markerfacecolor": BG,
            "markeredgewidth": 1.8,
        }
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.8,
        label=label,
        zorder=zorder,
        **marker_args,
    )


def add_value_label(ax, x, y, text, color, xytext=(0, 14), ha="center"):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=10,
        fontweight="semibold",
        color=color,
        bbox={"facecolor": BG, "edgecolor": GRID, "boxstyle": "round,pad=0.3"},
    )


def smooth_histogram(values, bins, sigma=2.0):
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    radius = max(3, int(sigma * 4))
    kernel_x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(kernel_x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    return centers, smooth


def print_results(base, strat, base_ev, strat_ev, base_price, strat_price):
    base_rev, base_op, _, base_fcf = base
    strat_rev, strat_op, _, strat_fcf = strat

    base_rev_2029 = b(p50(base_rev[:, -1]))
    strat_rev_2029 = b(p50(strat_rev[:, -1]))
    base_profit_2029 = b(p50(base_op[:, -1]))
    strat_profit_2029 = b(p50(strat_op[:, -1]))
    base_fcf_2029 = b(p50(base_fcf[:, -1]))
    strat_fcf_2029 = b(p50(strat_fcf[:, -1]))
    base_ev_p50 = p50(base_ev)
    strat_ev_p50 = p50(strat_ev)

    print("\nFINAL 5-YEAR RESULTS (2029, median / P50)")
    print("=" * 64)
    print(
        f"Revenue:          baseline ${base_rev_2029:6.2f}B | "
        f"strategy ${strat_rev_2029:6.2f}B | "
        f"impact +${strat_rev_2029 - base_rev_2029:5.2f}B "
        f"({pct_change(strat_rev_2029, base_rev_2029):.1f}%)"
    )
    print(
        f"Operating profit: baseline ${base_profit_2029:6.2f}B | "
        f"strategy ${strat_profit_2029:6.2f}B | "
        f"impact +${strat_profit_2029 - base_profit_2029:5.2f}B "
        f"({pct_change(strat_profit_2029, base_profit_2029):.1f}%)"
    )
    print(
        f"FCF:              baseline ${base_fcf_2029:6.2f}B | "
        f"strategy ${strat_fcf_2029:6.2f}B | "
        f"impact +${strat_fcf_2029 - base_fcf_2029:5.2f}B "
        f"({pct_change(strat_fcf_2029, base_fcf_2029):.1f}%)"
    )
    print(
        f"Enterprise value: baseline ${base_ev_p50:6.2f}B | "
        f"strategy ${strat_ev_p50:6.2f}B | "
        f"impact +${strat_ev_p50 - base_ev_p50:5.2f}B "
        f"({pct_change(strat_ev_p50, base_ev_p50):.1f}%)"
    )
    print(
        f"Value per share:  baseline ${p50(base_price):6.2f}  | "
        f"strategy ${p50(strat_price):6.2f}  | "
        f"impact +${p50(strat_price) - p50(base_price):5.2f}"
    )
    print("-" * 64)
    print(f"Revenue outperformance probability: {(strat_rev[:, -1] > base_rev[:, -1]).mean():.1%}")
    print(f"Profit outperformance probability:  {(strat_op[:, -1] > base_op[:, -1]).mean():.1%}")
    print(f"EV outperformance probability:      {(strat_ev > base_ev).mean():.1%}")


# -----------------------------
# Graphs
# -----------------------------

def save_revenue_graph(base_rev, strat_rev):
    fig, ax = plt.subplots(figsize=(13, 7.4))
    setup_axis(ax)
    x = np.arange(N_YEARS + 1)
    labels = ["2024", "2025", "2026", "2027", "2028", "2029"]

    base_med = np.concatenate([[ML_PARAMS["initial_revenue"]], np.median(base_rev, axis=0)]) / 1000
    strat_med = np.concatenate([[ML_PARAMS["initial_revenue"]], np.median(strat_rev, axis=0)]) / 1000
    base_p10 = np.concatenate([[ML_PARAMS["initial_revenue"]], np.percentile(base_rev, 10, axis=0)]) / 1000
    base_p90 = np.concatenate([[ML_PARAMS["initial_revenue"]], np.percentile(base_rev, 90, axis=0)]) / 1000
    strat_p10 = np.concatenate([[ML_PARAMS["initial_revenue"]], np.percentile(strat_rev, 10, axis=0)]) / 1000
    strat_p90 = np.concatenate([[ML_PARAMS["initial_revenue"]], np.percentile(strat_rev, 90, axis=0)]) / 1000

    ax.fill_between(x, base_p10, base_p90, color=BASELINE_FILL, alpha=0.13, label="Baseline P10-P90")
    ax.fill_between(x, strat_p10, strat_p90, color=RED, alpha=0.12, label="Strategy P10-P90")
    glow_line(ax, x, base_med, BASELINE, "Baseline median")
    glow_line(ax, x, strat_med, RED, "Strategy median")

    uplift = strat_med[-1] - base_med[-1]
    add_value_label(ax, x[-1], strat_med[-1], f"${strat_med[-1]:.1f}B", RED, xytext=(14, 4), ha="left")
    add_value_label(ax, x[-1], base_med[-1], f"${base_med[-1]:.1f}B", BASELINE, xytext=(14, -22), ha="left")
    ax.text(
        0.63,
        0.78,
        f"+${uplift:.1f}B revenue uplift",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="semibold",
        color=RED,
        bbox={
            "facecolor": "#FFF9EF",
            "edgecolor": RED,
            "linewidth": 1.1,
            "boxstyle": "round,pad=0.40",
        },
    )

    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"${val:.0f}B"))
    ax.set_title("Revenue Growth Paths", pad=18)
    ax.set_ylabel("Revenue")
    ax.set_ylim(0, max(strat_p90) * 1.12)
    ax.legend(ncol=2, loc="upper left")
    fig.suptitle("Monte Carlo Simulation  |  10,000 Runs", fontsize=18, fontweight="semibold", color=BROWN, y=0.98)
    fig.tight_layout()
    fig.savefig("revenue_paths.png", dpi=300)
    plt.close(fig)


def save_profit_graph(base_op, strat_op):
    fig, ax = plt.subplots(figsize=(14, 7.8))
    setup_axis(ax)
    base_y5 = base_op[:, -1] / 1000
    strat_y5 = strat_op[:, -1] / 1000
    bins = np.linspace(
        min(base_y5.min(), strat_y5.min()),
        max(base_y5.max(), strat_y5.max()),
        78,
    )
    base_x, base_density = smooth_histogram(base_y5, bins, sigma=1.7)
    strat_x, strat_density = smooth_histogram(strat_y5, bins, sigma=1.9)
    density_scale = 1000

    ax.hist(
        base_y5,
        bins=bins,
        density=True,
        color=BASELINE_FILL,
        alpha=0.18,
        edgecolor=EDGE,
        linewidth=0.45,
    )
    ax.hist(
        strat_y5,
        bins=bins,
        density=True,
        color=RED,
        alpha=0.20,
        edgecolor=EDGE,
        linewidth=0.45,
    )

    for x_vals, y_vals, color in [
        (base_x, base_density * density_scale, BASELINE),
        (strat_x, strat_density * density_scale, RED),
    ]:
        ax.fill_between(x_vals, y_vals, color=color, alpha=0.18, zorder=3)
        for linewidth, alpha in [(12, 0.04), (7, 0.08), (3.5, 0.18)]:
            ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
        ax.plot(x_vals, y_vals, color=color, linewidth=3.0, zorder=5)

    ax.hist(
        base_y5,
        bins=bins,
        density=True,
        weights=np.ones_like(base_y5) * density_scale,
        histtype="step",
        color=BASELINE,
        linewidth=1.0,
        alpha=0.45,
    )
    ax.hist(
        strat_y5,
        bins=bins,
        density=True,
        weights=np.ones_like(strat_y5) * density_scale,
        histtype="step",
        color=RED,
        linewidth=1.0,
        alpha=0.45,
    )

    y_top = max(base_density.max(), strat_density.max()) * density_scale * 1.30
    base_median = np.median(base_y5)
    strat_median = np.median(strat_y5)
    for median, color in [
        (base_median, BASELINE),
        (strat_median, RED),
    ]:
        for linewidth, alpha in [(9, 0.06), (5, 0.14), (2, 1.0)]:
            ax.axvline(median, color=color, lw=linewidth, alpha=alpha, linestyle="--")

    ax.text(
        base_median,
        y_top * 0.91,
        f"Baseline Median\n${base_median:.2f}B",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="semibold",
        color=BASELINE,
        bbox={
            "facecolor": BG,
            "edgecolor": BASELINE,
            "linewidth": 1.2,
            "boxstyle": "round,pad=0.40",
        },
    )
    ax.text(
        strat_median + 0.55,
        y_top * 0.68,
        f"Strategy Median\n${strat_median:.2f}B",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="semibold",
        color=RED,
        bbox={
            "facecolor": BG,
            "edgecolor": RED,
            "linewidth": 1.2,
            "boxstyle": "round,pad=0.40",
        },
    )

    profit_uplift = np.median(strat_y5) - np.median(base_y5)
    win_rate = (strat_y5 > base_y5).mean()
    result_rows = [
        ("Median Lift", f"+${profit_uplift:.2f}B"),
        ("Win Rate", f"{win_rate:.1%}"),
    ]
    card_text = "\n".join(f"{label:<12} {value:>8}" for label, value in result_rows)
    ax.text(
        0.965,
        0.90,
        card_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="semibold",
        color=BROWN,
        linespacing=1.55,
        family=FONT_FAMILY[0],
        bbox={
            "facecolor": "#FFF9EF",
            "edgecolor": GRID,
            "linewidth": 1.1,
            "boxstyle": "round,pad=0.60",
        },
    )
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"${val:.1f}B"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"{val / density_scale:.2f}"))
    ax.set_title("Year 5 Operating Profit Distribution", pad=18)
    ax.set_xlabel("2029 Operating profit")
    ax.set_ylabel("Relative density")
    ax.set_ylim(0, y_top)
    ax.set_xlim(min(base_y5.min(), strat_y5.min()) - 0.25, max(base_y5.max(), strat_y5.max()) + 0.40)
    fig.suptitle(
        "Profit Distribution  |  Baseline vs Strategy",
        fontsize=20,
        fontweight="semibold",
        color=BROWN,
        y=0.982,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("profit_distribution.png", dpi=300)
    plt.close(fig)


def save_ev_graph(base_ev, strat_ev, base_price, strat_price, base_fcf, strat_fcf):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.2), gridspec_kw={"wspace": 0.32})
    fig.suptitle("DCF Valuation Bridge", fontsize=22, fontweight="semibold", color=BROWN, y=0.985)

    ax = axes[0]
    setup_axis(ax)
    values = [np.median(base_ev), np.median(strat_ev)]
    y_positions = [0, 1]
    bar_h = 0.46
    draw_3d_barh(ax, y_positions[0], values[0], bar_h, BASELINE, "#4F2518", BASELINE_LIGHT, EDGE)
    draw_3d_barh(ax, y_positions[1], values[1], bar_h, RED, "#7A261F", RED_LIGHT, EDGE)

    ax.text(values[0] + 2.5, y_positions[0], f"${values[0]:.1f}B", va="center", color=BASELINE, fontweight="semibold", fontsize=12)
    ax.text(values[1] + 2.5, y_positions[1], f"${values[1]:.1f}B", va="center", color=RED, fontweight="semibold", fontsize=12)

    uplift = values[1] - values[0]
    uplift_pct = pct_change(values[1], values[0])
    y_arrow = 1.43
    ax.annotate(
        "",
        xy=(values[1], y_arrow),
        xytext=(values[0], y_arrow),
        arrowprops={"arrowstyle": "<->", "color": BASELINE_LIGHT, "lw": 1.8},
    )
    ax.text(
        (values[0] + values[1]) / 2,
        y_arrow + 0.07,
        f"+${uplift:.1f}B  (+{uplift_pct:.0f}%)",
        ha="center",
        va="bottom",
        color=BASELINE,
        fontweight="semibold",
        fontsize=11,
    )
    ax.text(
        0.02,
        0.94,
        f"Value/share: baseline \\${np.median(base_price):.2f}, strategy \\${np.median(strat_price):.2f}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox={"facecolor": BG, "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )

    ax.set_yticks(y_positions, ["Baseline\nEnterprise Value", "Strategy\nEnterprise Value"])
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_ylim(-0.55, 1.75)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"${val:.0f}B"))
    ax.set_title("Enterprise Value Comparison", fontsize=13, pad=14)
    ax.grid(True, axis="x", color=GRID, alpha=0.30, linewidth=0.6)
    ax.grid(False, axis="y")

    ax = axes[1]
    setup_axis(ax)
    x = np.arange(N_YEARS + 1)
    labels = ["2024", "2025", "2026", "2027", "2028", "2029"]
    fcf_b_med = np.concatenate([[ML_PARAMS["initial_fcf"]], np.median(base_fcf, axis=0)]) / 1000
    fcf_s_med = np.concatenate([[ML_PARAMS["initial_fcf"]], np.median(strat_fcf, axis=0)]) / 1000
    fcf_b_p10 = np.concatenate([[ML_PARAMS["initial_fcf"]], np.percentile(base_fcf, 10, axis=0)]) / 1000
    fcf_b_p90 = np.concatenate([[ML_PARAMS["initial_fcf"]], np.percentile(base_fcf, 90, axis=0)]) / 1000
    fcf_s_p10 = np.concatenate([[ML_PARAMS["initial_fcf"]], np.percentile(strat_fcf, 10, axis=0)]) / 1000
    fcf_s_p90 = np.concatenate([[ML_PARAMS["initial_fcf"]], np.percentile(strat_fcf, 90, axis=0)]) / 1000

    ax.fill_between(x, fcf_b_p10, fcf_b_p90, color=BASELINE_FILL, alpha=0.13)
    ax.fill_between(x, fcf_s_p10, fcf_s_p90, color=RED, alpha=0.12)
    glow_line(ax, x, fcf_b_med, BASELINE, "Baseline FCF")
    glow_line(ax, x, fcf_s_med, RED, "Strategy FCF")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"${val:.1f}B"))
    ax.set_title("Free Cash Flow Trajectory", fontsize=13, pad=14)
    ax.legend(loc="upper left")
    ax.set_ylim(max(0, min(fcf_b_p10.min(), fcf_s_p10.min()) * 0.75), max(fcf_s_p90) * 1.08)
    ax.grid(True, color=GRID, alpha=0.30, linewidth=0.6)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.12, wspace=0.32)
    fig.savefig("enterprise_value_bridge.png", dpi=300)
    plt.close(fig)


def save_summary_graph(base, strat, base_ev, strat_ev):
    base_rev, base_op, _, base_fcf = base
    strat_rev, strat_op, _, strat_fcf = strat

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Chipotle Monte Carlo + DCF Results", fontsize=22, fontweight="semibold", color=BROWN, y=0.992)
    fig.text(
        0.5,
        0.948,
        "2029 median outcomes from 10,000 simulations, with DCF valuation impact",
        ha="center",
        fontsize=12,
        color=BROWN,
    )

    metrics = ["Revenue", "Profit", "FCF", "EV"]
    baseline = [
        b(np.median(base_rev[:, -1])),
        b(np.median(base_op[:, -1])),
        b(np.median(base_fcf[:, -1])),
        np.median(base_ev),
    ]
    strategy = [
        b(np.median(strat_rev[:, -1])),
        b(np.median(strat_op[:, -1])),
        b(np.median(strat_fcf[:, -1])),
        np.median(strat_ev),
    ]

    ax = axes[0, 0]
    setup_axis(ax)
    x = np.arange(len(metrics))
    width = 0.38
    base_bars = ax.bar(
        x - width / 2,
        baseline,
        width=width,
        color=BASELINE,
        edgecolor=EDGE,
        linewidth=1.0,
        alpha=0.95,
        label="Baseline",
    )
    strat_bars = ax.bar(
        x + width / 2,
        strategy,
        width=width,
        color=RED,
        edgecolor=EDGE,
        linewidth=1.0,
        alpha=0.95,
        label="Strategy",
    )
    for bars in [base_bars, strat_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(strategy) * 0.018,
                f"${height:.1f}B",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color=BROWN,
            )
    ax.set_xticks(x, metrics)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"${val:.0f}B"))
    ax.set_title("2029 / Valuation Medians")
    ax.legend()
    ax.set_ylim(0, max(strategy) * 1.18)
    ax.grid(True, axis="y")

    ax = axes[0, 1]
    setup_axis(ax)
    uplift = np.median(strat_rev, axis=0) / 1000 - np.median(base_rev, axis=0) / 1000
    bottoms = np.zeros(N_YEARS)
    for solution, color in zip(SOLUTIONS.values(), [RED, GOLD, BROWN]):
        values = uplift * solution["weight"]
        ax.bar(
            YEARS,
            values,
            bottom=bottoms,
            color=color,
            edgecolor=EDGE,
            linewidth=0.9,
            label=solution["name"],
        )
        bottoms += values
    for year, total in zip(YEARS, bottoms):
        ax.text(year, total + max(bottoms) * 0.035, f"${total:.1f}B", ha="center", fontweight="semibold", fontsize=10)
    ax.set_title("Revenue Uplift Attribution")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"${val:.1f}B"))
    ax.grid(True, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    setup_axis(ax)
    thresholds = np.linspace(0, 8, 300)
    profit_advantage = (strat_op[:, -1] - base_op[:, -1]) / 1000
    probs = [(profit_advantage > threshold).mean() for threshold in thresholds]
    glow_line(ax, thresholds, probs, RED, markers=False)
    ax.fill_between(thresholds, probs, color=RED, alpha=0.12)
    ax.axhline((profit_advantage > 0).mean(), color=BASELINE_LIGHT, linestyle="--")
    ax.text(
        0.04,
        0.91,
        f"{(profit_advantage > 0).mean():.1%} chance strategy profit beats baseline",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="semibold",
        color=RED,
        bbox={"facecolor": BG, "edgecolor": GRID, "boxstyle": "round,pad=0.4"},
    )
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"${val:.0f}B"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda val, _: f"{val:.0%}"))
    ax.set_title("Probability of Profit Advantage")
    ax.set_xlabel("Strategy profit advantage")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True)

    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ["Revenue impact", f"+${strategy[0] - baseline[0]:.2f}B"],
        ["Profit impact", f"+${strategy[1] - baseline[1]:.2f}B"],
        ["FCF impact", f"+${strategy[2] - baseline[2]:.2f}B"],
        ["EV impact", f"+${strategy[3] - baseline[3]:.2f}B"],
        ["Profit win rate", f"{(strat_op[:, -1] > base_op[:, -1]).mean():.1%}"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Result"],
        loc="center",
        cellLoc="left",
        colWidths=[0.52, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(BROWN)
            cell.set_text_props(color="white", fontweight="semibold")
        else:
            cell.set_facecolor(BG)
            cell.set_text_props(color=BROWN, fontweight="semibold" if col == 1 else "normal")
    ax.set_title("Final Effects", fontweight="semibold")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("final_results_summary.png", dpi=300)
    plt.close(fig)


def main():
    print("Running 10,000-simulation Monte Carlo model...")
    baseline = run_baseline()
    strategy = run_strategy("base")

    baseline_fcf_for_valuation = fixed_baseline_fcf()
    base_ev, _, base_price = dcf_enterprise_value(baseline_fcf_for_valuation)
    strat_ev, _, strat_price = dcf_enterprise_value(strategy[3])

    print_results(baseline, strategy, base_ev, strat_ev, base_price, strat_price)

    save_revenue_graph(baseline[0], strategy[0])
    save_profit_graph(baseline[1], strategy[1])
    save_ev_graph(base_ev, strat_ev, base_price, strat_price, baseline[3], strategy[3])
    save_summary_graph(baseline, strategy, base_ev, strat_ev)

    print("\nGraphs saved:")
    print("  revenue_paths.png")
    print("  profit_distribution.png")
    print("  enterprise_value_bridge.png")
    print("  final_results_summary.png")


if __name__ == "__main__":
    main()
