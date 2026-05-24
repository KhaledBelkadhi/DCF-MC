import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import time

# ==========================================
# PAGE CONFIGURATION & BRANDING
# ==========================================
st.set_page_config(page_title="Advanced MC Valuation", layout="wide", page_icon="📊")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942269.png", width=60)
st.sidebar.markdown("### **South Mediterranean University (SMU)**")
st.sidebar.markdown("**MSc Quantitative Finance**")
st.sidebar.markdown("---")

st.title("📊 Advanced Monte Carlo DCF Valuation")
st.write("Watch the intrinsic value distribution build in real-time while monitoring the Delta Convergence algorithm (stabilizing on Mean Price).")

# ==========================================
# SIDEBAR: BASE VALUATION INPUTS
# ==========================================
st.sidebar.header("1. Firm Fundamentals")
current_price = st.sidebar.number_input("Current Market Price ($)", value=45.00, step=1.0)
shares_out = st.sidebar.number_input("Shares Outstanding (M)", value=100)
net_debt = st.sidebar.number_input("Net Debt ($M)", value=500)

n_years = st.sidebar.slider("Forecast Horizon (Years)", 3, 10, 5)

st.sidebar.markdown("**Base FCF Generation ($M):**")
fcf_start = st.sidebar.number_input("Year 1 FCF", value=100.0)
fcf_growth = st.sidebar.number_input("Annual FCF Growth (%)", value=5.0) / 100
fcff_list = [fcf_start * (1 + fcf_growth)**i for i in range(n_years)]

st.sidebar.header("2. Simulation Engine")
mc_type = st.sidebar.radio("Monte Carlo Method:", ["Parametric (Normal Dist)", "Bootstrapping (Historical Resampling)"])

st.sidebar.markdown("---")
# ==========================================
# SIDEBAR: METHOD-SPECIFIC INPUTS
# ==========================================
if mc_type == "Parametric (Normal Dist)":
    st.sidebar.markdown("**Parametric Assumptions**")
    wacc_mean = st.sidebar.slider("WACC Mean (%)", 5.0, 15.0, 8.0, 0.5) / 100
    wacc_std = st.sidebar.slider("WACC Std Dev (%)", 0.0, 5.0, 1.0, 0.1) / 100
    g_mean = st.sidebar.slider("Terminal Growth 'g' Mean (%)", 0.0, 5.0, 2.0, 0.5) / 100
    g_std = st.sidebar.slider("Growth Std Dev (%)", 0.0, 3.0, 0.5, 0.1) / 100

elif mc_type == "Bootstrapping (Historical Resampling)":
    st.sidebar.markdown("**Historical Data Generator**")
    crisis_prob = st.sidebar.slider("Probability of Market Crisis (%)", 0, 50, 20) / 100
    
    np.random.seed(42)
    hist_size = 2000 
    is_crisis = np.random.binomial(1, crisis_prob, hist_size)
    hist_wacc = np.where(is_crisis, np.random.normal(0.12, 0.02, hist_size), np.random.normal(0.08, 0.01, hist_size))
    hist_g = np.where(is_crisis, np.random.normal(0.00, 0.01, hist_size), np.random.normal(0.025, 0.005, hist_size))

st.sidebar.markdown("---")
st.sidebar.header("3. Convergence Controls")
max_sims = st.sidebar.slider("Max Simulations (N)", 500, 50000, 5000, 500)
check_interval = st.sidebar.number_input("Check Convergence Every (X sims)", value=200)
delta_stop = st.sidebar.slider("Mean Delta Stop Threshold (%)", 0.1, 5.0, 0.5, 0.1) / 100
anim_speed = st.sidebar.slider("Animation Delay (ms)", 0, 500, 50, 50)

run_btn = st.sidebar.button("🚀 Execute Engine", type="primary")

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.markdown("### Projected Operating Cash Flows")
cols = st.columns(n_years)
for i, cash in enumerate(fcff_list):
    cols[i].metric(label=f"Year {i+1}", value=f"${cash:.1f}M")
st.markdown("---")

metrics_row = st.empty()
live_chart_placeholder = st.empty()

if run_btn:
    progress_bar = st.progress(0)
    
    stock_prices = []
    
    running_deltas = []
    check_points_delta = []
    wacc_history = []
    g_history = []
    
    buy_count = 0
    previous_mean = None
    converged = False
    final_sim_count = max_sims
    
    # NEW: Patience counter for robust convergence
    consecutive_stable_count = 0 
    
    cash_noise_matrix = np.random.normal(1, 0.05, (max_sims, n_years))
    start_time = time.time()
    
    for i in range(1, max_sims + 1):
        if mc_type == "Parametric (Normal Dist)":
            wacc_sim = np.random.normal(wacc_mean, wacc_std)
            g_sim = np.random.normal(g_mean, g_std)
        else: 
            wacc_sim = np.random.choice(hist_wacc)
            g_sim = np.random.choice(hist_g)
            
        if wacc_sim <= g_sim:
            wacc_sim = g_sim + 0.005 
            
        wacc_history.append(wacc_sim)
        g_history.append(g_sim)
            
        noise_cash = [fcff_list[yr] * cash_noise_matrix[i-1][yr] for yr in range(n_years)]
        tv = (noise_cash[-1] * (1 + g_sim)) / (wacc_sim - g_sim)
        
        cash_flows = [0] + noise_cash
        cash_flows[-1] += tv 
        ev = npf.npv(wacc_sim, cash_flows)
        
        price = (ev - net_debt) / shares_out
        stock_prices.append(price)
        
        if price > current_price * 1.05:
            buy_count += 1
            
        # LIVE UPDATE BLOCK
        if i % check_interval == 0:
            current_mean = np.mean(stock_prices)
            
            if previous_mean is not None:
                delta = abs(current_mean - previous_mean) / previous_mean
                running_deltas.append(delta * 100) 
                check_points_delta.append(i)
                
                # --- NEW PATIENCE LOGIC ---
                # 1. Require a dynamic minimum of runs (e.g., 20% of max_sims, or at least 500)
                min_sims_to_check = max(500, max_sims // 5)
                
                if i >= min_sims_to_check:
                    if delta < delta_stop:
                        consecutive_stable_count += 1
                    else:
                        consecutive_stable_count = 0 # Reset patience if it spikes again
                        
                    # 2. Must be completely stable for 3 checks in a row
                    if consecutive_stable_count >= 3:
                        converged = True
                        final_sim_count = i
                        break
                
            progress_bar.progress(i / max_sims)
            
            fig_live, (ax_delta, ax_hist) = plt.subplots(1, 2, figsize=(15, 4))
            
            if len(running_deltas) > 0:
                ax_delta.plot(check_points_delta, running_deltas, color='#FF9800', linewidth=2)
                ax_delta.axhline(y=delta_stop * 100, color='red', linestyle='--', label=f'Stop Threshold ({delta_stop*100:.2f}%)')
                ax_delta.legend()
                
            ax_delta.set_title(f"Live Delta Convergence (Current Mean: ${current_mean:.2f})")
            ax_delta.set_xlim(0, max_sims)
            
            if len(running_deltas) > 5:
                ax_delta.set_ylim(0, max(max(running_deltas[-5:]), delta_stop * 100) * 2)
            else:
                ax_delta.set_ylim(0, 5)
                
            ax_delta.set_xlabel("Simulations Run")
            ax_delta.set_ylabel("Mean Delta Change (%)")
            ax_delta.grid(True, linestyle=':', alpha=0.6)
            
            if len(stock_prices) > 10:
                p99 = np.percentile(stock_prices, 99)
                p01 = np.percentile(stock_prices, 1)
                filtered_live = [p for p in stock_prices if p01 <= p <= p99]
                ax_hist.hist(filtered_live, bins=40, color='#3F51B5', alpha=0.7, edgecolor='white')
                ax_hist.axvline(current_mean, color='green', linestyle='dashed', linewidth=2, label=f'Mean (${current_mean:.2f})')
                
            ax_hist.axvline(current_price, color='red', linestyle='dashed', linewidth=2, label='Market Price')
            ax_hist.set_title(f"Live Intrinsic Value Spread (N={i:,})")
            ax_hist.set_xlabel("Price Per Share ($)")
            ax_hist.set_ylabel("Frequency")
            ax_hist.legend()
            
            live_chart_placeholder.pyplot(fig_live)
            plt.close(fig_live)
            
            if anim_speed > 0:
                time.sleep(anim_speed / 1000.0)
            
            previous_mean = current_mean
    st.markdown("---")
    st.subheader("🔎 Stochastic Input Diagnostics")
    
    fig_hist, (ax_wacc, ax_g) = plt.subplots(1, 2, figsize=(15, 4))
    
    ax_wacc.hist(np.array(wacc_history) * 100, bins=40, color='#FF9800', alpha=0.7, edgecolor='white')
    ax_wacc.set_title("Sampled WACC Distribution (%)")
    
    ax_g.hist(np.array(g_history) * 100, bins=40, color='#00BCD4', alpha=0.7, edgecolor='white')
    ax_g.set_title("Sampled Terminal Growth Distribution (%)")
    exec_time = time.time() - start_time
    
    # Force progress bar to 100% when done
    progress_bar.progress(1.0)
    
    if converged:
        st.success(f"✅ **Convergence Reached!** Engine halted dynamically at **{final_sim_count:,} iterations**. Runtime: {exec_time:.2f}s.")
    else:
        st.warning(f"⚠️ **Max Simulations Hit.** Engine ran all {max_sims:,} iterations without fully stabilizing. Runtime: {exec_time:.2f}s.")
    
    # ==========================================
    # FINAL METRICS & INPUT DISTRIBUTIONS
    # ==========================================
    actual_sims = len(stock_prices)
    stock_prices = np.array(stock_prices)
    
    hold_cases = np.sum((stock_prices >= current_price * 0.95) & (stock_prices <= current_price * 1.05))
    sell_cases = np.sum(stock_prices < current_price * 0.95)
    
    p_buy = buy_count / actual_sims
    p_hold = hold_cases / actual_sims
    p_sell = sell_cases / actual_sims
    final_mean = np.mean(stock_prices)
    
    c1, c2, c3, c4 = metrics_row.columns(4)
    c1.metric("Mean Intrinsic Value", f"${final_mean:.2f}")
    c2.metric("BUY Probability", f"{p_buy * 100:.1f}%")
    c3.metric("HOLD Probability", f"{p_hold * 100:.1f}%")
    c4.metric("SELL Probability", f"{p_sell * 100:.1f}%")
    
    
    
    st.pyplot(fig_hist)