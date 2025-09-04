import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class ResultsHandler:
    def __init__(self, tickers, w_opt, w_old, w_benchmark, mu, cov, prices):
        self.tickers = tickers
        self.w_opt = pd.Series(w_opt, index=tickers, name="Optimized")
        self.w_old = w_old
        self.w_benchmark = w_benchmark
        self.mu = mu
        self.cov = cov
        self.prices = prices

    # -------------------- Portfolio Stats --------------------
    def portfolio_stats(self) -> dict:
        exp_ret = float(self.mu.values @ self.w_opt.values)
        vol = float(np.sqrt(self.w_opt.values.T @ self.cov.values @ self.w_opt.values))
        sharpe = exp_ret / vol if vol != 0 else 0
        return {
            "Expected Annual Return": exp_ret,
            "Annual Volatility": vol,
            "Sharpe Ratio": sharpe
        }

    # -------------------- Interactive Charts --------------------
    def plot_weights(self, st=None):
        """Compare Current, Optimized, and Benchmark Weights (interactive bar chart)."""
        df = pd.DataFrame({
            "Ticker": self.tickers,
            "Current": self.w_old.values,
            "Optimized": self.w_opt.values,
            "Benchmark": self.w_benchmark.values,
        })

        df_melt = df.melt(id_vars="Ticker", var_name="Portfolio", value_name="Weight")

        fig = px.bar(
            df_melt,
            x="Ticker",
            y="Weight",
            color="Portfolio",
            barmode="group",
            title="Comparison of Portfolio Weights",
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            bargap=0.3,
            bargroupgap=0.2
        )

        if st:
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig.show()

    def plot_return_vs_weight(self, st=None, top_n=15):
        """Optimized Weights vs Expected Returns (interactive bubble chart)."""
        df = pd.DataFrame({
            "Ticker": self.tickers,
            "Optimized Weight": self.w_opt.values,
            "Expected Return": self.mu.values,
        })

        # Keep only positive weights
        df = df[df["Optimized Weight"] > 0]

        # Take top N by weight
        df = df.sort_values("Optimized Weight", ascending=False).head(top_n)

        # Plot bubble chart
        fig = px.scatter(
            df,
            x="Optimized Weight",
            y="Expected Return",
            size="Optimized Weight",
            color="Expected Return",
            text="Ticker",
            hover_data={
                "Optimized Weight": ":.2%",
                "Expected Return": ":.2%",
                "Ticker": True
            },
            color_continuous_scale="Blues",
            title="Optimized Weights vs Expected Returns",
        )

        # Place ticker labels cleanly
        fig.update_traces(textposition="top center")

        # Layout improvements
        fig.update_layout(
            xaxis_title="Optimized Weight",
            yaxis_title="Expected Return",
            template="plotly_dark",
            height=600,
            margin=dict(l=60, r=60, t=80, b=60),
        )

        if st:
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig.show()

    def plot_portfolio_vs_benchmark(self, st=None):
        """Portfolio vs Benchmark performance (interactive line chart)."""
        returns = self.prices.pct_change().dropna()
        port_ret = returns @ self.w_opt
        bench_ret = returns @ self.w_benchmark

        df = pd.DataFrame({
            "Optimized Portfolio": (1 + port_ret).cumprod(),
            "Benchmark (NIFTY 50)": (1 + bench_ret).cumprod(),
        })

        fig = px.line(
            df,
            x=df.index,
            y=df.columns,
            title="Growth of Portfolio vs Benchmark",
        )

        #  ---- Added Proper Axis Labels
        fig.update_layout(
            xaxis_title="Time (Trading Days)",
            yaxis_title="Portfolio Value (Growth of ₹1)",
            margin=dict(l=80, r=80, t=80, b=80),
            xaxis=dict(tickmode='auto'),
        )

        if st:
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig.show()

    def plot_allocation_pie_chart(self, st=None, top_n=15):
        """Donut charts for Current, Optimized, and Benchmark portfolios."""
        def prepare_weights(weights, top_n):
            weights = weights[weights > 0]
            if len(weights) > top_n:
                top = weights.nlargest(top_n)
                others = pd.Series([weights.drop(top.index).sum()], index=["Others"])
                return pd.concat([top, others])
            return weights

        portfolios = {
            "📊 Current Portfolio": prepare_weights(self.w_old, top_n),
            "🚀 Optimized Portfolio": prepare_weights(self.w_opt, top_n),
            "🏦 Benchmark (NIFTY 50)": prepare_weights(self.w_benchmark, top_n),
        }

        if st:
            tabs = st.tabs(list(portfolios.keys()))
            for tab, (name, weights) in zip(tabs, portfolios.items()):
                with tab:
                    df = pd.DataFrame({"Stock": weights.index, "Weight": weights.values})
                    fig = px.pie(
                        df,
                        names="Stock",
                        values="Weight",
                        hole=0.4,
                        title=name,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_traces(textinfo="percent+label")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            for name, weights in portfolios.items():
                df = pd.DataFrame({"Stock": weights.index, "Weight": weights.values})
                fig = px.pie(
                    df,
                    names="Stock",
                    values="Weight",
                    hole=0.4,
                    title=name,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.show()

    def build_results_table(self):
        """Builds a DataFrame comparing weights and expected returns for each ticker."""
        df = pd.DataFrame({
            "Ticker": self.tickers,
            "Current Weight": self.w_old.values,
            "Optimized Weight": self.w_opt.values,
            "Benchmark Weight": self.w_benchmark.values,
            "Expected Return": self.mu.values
        })
        return df
