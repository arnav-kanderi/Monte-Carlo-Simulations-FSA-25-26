import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Colors
chipotle_red = "#A32020"
chipotle_brown = "#451400"
chipotle_gold = "#E2A100"
chipotle_green = "#4E7C59"

#numbers

initial_revenue = 14.3  
baseline_margin = 0.169  

years = 5
simulations = 10000

baseline_profit = []
strategy_profit = []

baseline_paths = []
strategy_paths = []

# Montecarlo Logic

for sim in range(simulations):

    revenue_base = initial_revenue
    revenue_strategy = initial_revenue

    base_path = [revenue_base]
    strat_path = [revenue_strategy]

    for year in range(years):

        # Baseline
        base_growth = np.random.normal(0.145, 0.015)

        # Strategy
        international_boost = np.random.triangular(0.005, 0.01, 0.02)   
        retail_boost = np.random.triangular(0.002, 0.006, 0.012)       
        ventures_boost = np.random.triangular(0.001, 0.003, 0.006)    

        strategy_growth = base_growth + international_boost + retail_boost + ventures_boost

        revenue_base *= (1 + base_growth)
        revenue_strategy *= (1 + strategy_growth)

        base_path.append(revenue_base)
        strat_path.append(revenue_strategy)

    baseline_paths.append(base_path)
    strategy_paths.append(strat_path)

    
    # Margins


    # Baseline margins
    margin_base = np.random.normal(baseline_margin, 0.008)

    # Strategy margins
    automation_savings = np.random.triangular(0.002, 0.004, 0.007)    
    supply_chain_savings = np.random.triangular(0.001, 0.003, 0.005) 
    forecasting_savings = np.random.triangular(0.001, 0.002, 0.004)  

    margin_strategy = margin_base + automation_savings + supply_chain_savings + forecasting_savings

    profit_base = revenue_base * margin_base
    profit_strat = revenue_strategy * margin_strategy

    baseline_profit.append(profit_base)
    strategy_profit.append(profit_strat)

baseline_profit = np.array(baseline_profit)
strategy_profit = np.array(strategy_profit)


probability = np.mean(strategy_profit > baseline_profit)
print(f"Probability strategy beats baseline: {probability:.2%}")


# VISUALIZATION SYSTEM



plt.style.use("default")


plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#451400'
plt.rcParams['axes.labelcolor'] = '#451400'
plt.rcParams['text.color'] = '#451400'
plt.rcParams['xtick.color'] = '#451400'
plt.rcParams['ytick.color'] = '#451400'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

difference = strategy_profit - baseline_profit
sorted_diff = np.sort(difference)
probabilities = 1 - (np.arange(len(sorted_diff)) / len(sorted_diff))

baseline_avg = np.mean(baseline_profit)
strategy_avg = np.mean(strategy_profit)
uplift = strategy_avg - baseline_avg

international_contrib = uplift * 0.35
retail_contrib = uplift * 0.25
automation_contrib = uplift * 0.25
supply_chain_contrib = uplift * 0.15

graphs = []

# GRAPH 1 — Profit Distribution

def graph_profit_distribution():
    plt.clf()

    plt.hist(baseline_profit, bins=60, alpha=0.65,
             color=chipotle_brown, label="Baseline")

    plt.hist(strategy_profit, bins=60, alpha=0.75,
             color=chipotle_red, label="Strategy")

    plt.axvline(np.mean(baseline_profit), color=chipotle_brown, linestyle='--', linewidth=2)
    plt.axvline(np.mean(strategy_profit), color=chipotle_red, linestyle='--', linewidth=2)

    plt.title("Monte Carlo Profit Projection (2025–2029)", fontsize=18, weight="bold")
    plt.xlabel("Projected Profit in Year 5 ($ Billions)")
    plt.ylabel("Frequency")

    plt.legend()
    plt.grid(alpha=0.15)

    plt.savefig("profit_distribution.png", dpi=300)


graphs.append(graph_profit_distribution)


# GRAPH 2 — Revenue Paths

def graph_revenue_paths():
    plt.clf()

    baseline_array = np.array(baseline_paths)
    strategy_array = np.array(strategy_paths)

    baseline_median = np.median(baseline_array, axis=0)
    strategy_median = np.median(strategy_array, axis=0)

    baseline_median[0] = initial_revenue
    strategy_median[0] = initial_revenue

    baseline_low = np.percentile(baseline_array, 25, axis=0)
    baseline_high = np.percentile(baseline_array, 75, axis=0)

    strategy_low = np.percentile(strategy_array, 25, axis=0)
    strategy_high = np.percentile(strategy_array, 75, axis=0)

    plt.fill_between(range(len(baseline_median)), baseline_low, baseline_high,
                     color=chipotle_brown, alpha=0.15)

    plt.fill_between(range(len(strategy_median)), strategy_low, strategy_high,
                     color=chipotle_gold, alpha=0.25)

    plt.plot(baseline_median, color=chipotle_brown, linewidth=3, label="Baseline Median")
    plt.plot(strategy_median, color=chipotle_red, linewidth=3, label="Strategy Median")

    plt.legend()

    plt.title("Simulated Revenue Growth Paths", fontsize=18, weight="bold")
    plt.xlabel("Years Ahead")
    plt.ylabel("Revenue ($ Billions)")

    plt.grid(alpha=0.2)

    plt.savefig("revenue_paths.png", dpi=300)


graphs.append(graph_revenue_paths)

# GRAPH 3 — Probability Curve

def graph_probability_curve():
    plt.clf()
    sorted_diff_extended = np.insert(sorted_diff, 0, 0)
    probabilities_extended = np.insert(probabilities, 0, probabilities[0])

    plt.plot(sorted_diff_extended, probabilities_extended, color=chipotle_red, linewidth=3)
    plt.fill_between(
        sorted_diff_extended,
        probabilities_extended,
        color=chipotle_red,
        alpha=0.3
    )
    plt.axvline(0, color=chipotle_brown, linestyle="--")

    plt.title("Probability Curve: Strategy Outperformance", fontsize=18, weight="bold")
    plt.xlabel("Profit Advantage of Strategy ($ Billions)")
    plt.ylabel("Probability Strategy ≥ Baseline")

    plt.xlim(left=0)

    plt.grid(alpha=0.2)
    plt.savefig("probability_curve.png", dpi=300)

graphs.append(graph_probability_curve)

# GRAPH 4 — Value Creation

def graph_value_creation():
    plt.clf()

    labels = [
        "Baseline Profit",
        "International Expansion",
        "Retail Products",
        "Automation + AI",
        "Supply Chain Optimization",
        "Total Strategy Profit"
    ]

    values = [
        baseline_avg,
        international_contrib,
        retail_contrib,
        automation_contrib,
        supply_chain_contrib,
        strategy_avg
    ]

    colors = [
        "#5A3E36",   
        "#E2A100",   
        "#D95F02",  
        "#A32020",  
        "#4E7C59",   
        "#e8dfd8"   
    ]

    plt.bar(labels, values, color=colors, edgecolor="#451400", linewidth=1.5)
    for i, v in enumerate(values):
        plt.text(i, v + max(values)*0.02, f"{v:.2f}", ha='center', fontsize=11)

    plt.title("Value Creation Breakdown", fontsize=18, weight="bold")

    plt.ylabel("Contribution to Profit ($ Billions)")

    plt.xticks(rotation=25)

    plt.grid(axis="y", alpha=0.2)

    plt.savefig("value_creation_breakdown.png", dpi=300)


graphs.append(graph_value_creation)


# PROFESSIONAL DEMO MODE


current_graph = 0
fig = plt.figure(figsize=(11,7))

manager = plt.get_current_fig_manager()
try:
    manager.full_screen_toggle()
except:
    try:
        manager.window.state("zoomed")
    except:
        pass

def update(frame):
    global current_graph
    plt.clf()
    graphs[current_graph]()
    current_graph = (current_graph + 1) % len(graphs)


ani = FuncAnimation(fig, update, interval=3000)

graphs[0]()

plt.show()