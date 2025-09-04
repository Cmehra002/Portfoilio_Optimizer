import streamlit as st
import numpy as np
import pandas as pd

from portfolio_optimizer.data_handler import DataHandler
from portfolio_optimizer.optimizer import PortfolioOptimizer
from portfolio_optimizer.results_handler import ResultsHandler  # already updated with Plotly

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("### 🎯 Portfolio Optimizer")

    st.text(
        "This tool helps you find the best way to split your money across stocks "
        "to balance risk and return. 📊"
    )

    # --- Collapsible Objectives ---
    with st.expander("💡 What are we trying to achieve? (Objectives)"):
        st.markdown("""
        - **📈 Grow Money (Maximize Returns)**  
          Try to put more money in stocks that are expected to perform better.  

        - **🛡️ Play Safe (Minimize Risk)**  
          Try to reduce ups and downs by spreading money more evenly.  

        **Example formulas**  
        `cp.Maximize(w @ mu)` → Grow money  
        `cp.Minimize(cp.quad_form(w, cov))` → Reduce risk
        """)

    # --- Collapsible Constraints ---
    with st.expander("📏 Rules to follow (Constraints)"):
        st.markdown("""
        - **💯 Fully Invested** → All money must be invested.  
          Formula: `cp.sum(w) == 1`

        - **🚫 No Short Selling** → You can’t bet against a stock.  
          Formula: `w >= 0`

        - **📊 Diversification Rule** → No stock gets more than 10%.  
          Formula: `w <= 0.1`

        - **🎯 Minimum Target Return** → Aim for at least 12% return per year.  
          Formula: `w @ mu >= 0.12`

        - **⚖️ Risk Limit** → Keep risk below a certain level.  
          Formula: `cp.quad_form(w, cov) <= 0.05`

        - **📌 Stay Close to Benchmark** → Don’t drift too far from NIFTY 50.  
          Formula: `cp.norm(w - w_benchmark, 1) <= 0.2`

        - **🔄 Limit Changes (Turnover)** → Avoid too much buying/selling.  
          Formula: `cp.norm(w - w_old, 1) <= 0.3`
        """)

# --- Main Layout ---
left_col, right_col = st.columns([3, 1])

with left_col:
    st.header("Welcome to Portfolio Optimizer 🌐", divider='blue')

    with st.expander("ℹ️ What do these terms mean?"):
        st.markdown("""
        - **w** → The weights (how much % money goes into each stock).  
        - **mu (μ)** → Expected yearly returns of each stock.  
        - **cov** → Risk matrix showing how stocks move together.  
        """)

    objectives = st.text_input(
        "Enter your investment objective 💸:",
        value="cp.Maximize(w @ mu)",
    )
    constraints = st.text_area(
        "Enter your investment constraints 💱:",
        value="cp.sum(w) == 1\nw >= 0\nw <= 0.1"
    )
    submit_bt = st.button("Optimize ☄️")

    if submit_bt:
        if not objectives or not constraints:
            st.error("Please fill in all fields to continue.")
        else:
            status_placeholder = st.empty()
            status_placeholder.info("Optimization in Progress…")
            progress_bar = st.progress(0)
            steps_box = st.empty()

            steps = [
                "📊 Step 1/4: Loading benchmark weights...",
                "💹 Step 2/4: Generating historical price data...",
                "📈 Step 3/4: Calculating returns & risk...",
                "⏳ Step 4/4: Running optimization..."
            ]

            def update_progress(step):
                percent = int((step + 1) / len(steps) * 100)
                progress_bar.progress(percent)
                steps_box.markdown(
                    "\n".join(
                        [f"{'✅' if i < step else '➡️' if i == step else '⬜'} {s}"
                         for i, s in enumerate(steps)]
                    )
                )

            # --- Step 1: Load Benchmark Weights ---
            update_progress(0)
            data_handler = DataHandler()
            nifty = data_handler.fetch_nifty50_composition()
            if nifty.empty:
                status_placeholder.error("❌ Failed to load benchmark weights. Please check your CSV file.")
            else:
                tickers = nifty['Symbol'].tolist()
                weights_benchmark = nifty['Weightage'].values

                # --- Step 2: Generate Price Data ---
                update_progress(1)
                np.random.seed(42)
                num_days = 252
                num_tickers = len(tickers)
                mean_return = 0.0003
                volatility = 0.02
                log_returns = np.random.normal(mean_return, volatility, (num_days, num_tickers))
                prices_data = 100 * np.exp(np.cumsum(log_returns, axis=0))
                prices_df = pd.DataFrame(prices_data, columns=tickers)

                # --- Step 3: Calculate Stats ---
                update_progress(2)
                mu, cov = data_handler.calculate_statistics(prices_df)
                w_old = weights_benchmark.copy()

                # --- Step 4: Run Optimization ---
                update_progress(3)
                constraints_list = [c.strip() for c in constraints.split("\n") if c.strip()]
                optimizer = PortfolioOptimizer(mu.values, cov.values, weights_benchmark, w_old)
                optimizer.define_problem(objectives, constraints_list)
                status, optimal_weights = optimizer.solve(solver='SCS')

                if status in ["optimal", "optimal_inaccurate"]:
                    result_handler = ResultsHandler(
                        tickers,
                        optimal_weights,
                        pd.Series(w_old, index=tickers),
                        pd.Series(weights_benchmark, index=tickers),
                        mu,
                        cov,
                        prices_df
                    )

                    # Portfolio Statistics (metric cards)
                    st.markdown("<hr style='border:2px solid #28a745;'>", unsafe_allow_html=True)
                    st.subheader("📈 Portfolio Statistics")
                    stats = result_handler.portfolio_stats()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("📈 Expected Return", f"{stats['Expected Annual Return']:.2%}")
                    col2.metric("📊 Annual Volatility", f"{stats['Annual Volatility']:.2%}")
                    col3.metric("⚖️ Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")

                    st.markdown("<hr style='border:2px solid #1E90FF;'>", unsafe_allow_html=True)

                    # --- Results ---
                    st.subheader("📊 Optimized Portfolio Weights and Expected Returns")
                    st.dataframe(result_handler.build_results_table())

                    # --- Interactive Charts ---
                    st.markdown("<hr style='border:2px solid #1E90FF;'>", unsafe_allow_html=True)
                    with st.expander("📊 Portfolio Weights Chart", expanded=True):
                        result_handler.plot_weights(st=st)


                    st.markdown("<hr style='border:2px solid #28a745;'>", unsafe_allow_html=True)
                    with st.expander("📈 Expected Return vs Portfolio Weight", expanded=True):
                        result_handler.plot_return_vs_weight(st=st)



                    st.markdown("<hr style='border:2px solid #ffc107;'>", unsafe_allow_html=True)
                    with st.expander("📉 Portfolio vs Benchmark Performance", expanded=True):
                        result_handler.plot_portfolio_vs_benchmark(st=st)


                    
                    st.markdown("<hr style='border:2px solid #28a745;'>", unsafe_allow_html=True)

                    with st.expander("🥧 Portfolio Allocation Donut Charts", expanded=True):
                        result_handler.plot_allocation_pie_chart(st=st, top_n=15)


          

                    # --- Mark Completion ---
                    steps_box.empty()  # This will remove the loading steps from the screen
                    progress_bar.progress(100)
                    status_placeholder.success("✅ Optimization Completed! 🎉")
                    st.markdown("<hr style='border:2px solid #ffc107;'>", unsafe_allow_html=True)
                    st.toast("🎉 Optimization completed successfully!")
                else:
                    status_placeholder.error(f"❌ Optimization failed: {status}")
                    st.snow()
