Quantitative Research Memo
------------------------

**Subject:** Optimization of Volcker Strategy for BTCUSDT - Addressing Adverse Selection using ML Alpha

Dear Futures First Quant Desk,

We are pleased to submit our analysis and optimization results for the Volume-Clocked Order Flow Imbalance (OFI) Market Maker strategy on BTCUSDT. Our goal was to improve upon the initial results, which featured an unfavorable performance due to adverse selection in trending moves. We have successfully addressed this issue by widening the spread and incorporating a Machine Learning (ML) Alpha scalar into our optimization process.

The raw tick data consisted of 37 million events compressed into 18,500 volume bars. We out-of-sample hit a 56.20% success rate using the original strategy parameters. However, upon closer examination, we realized that the unfavorable performance was caused primarily by adverse selection on trending moves. To mitigate this issue, we introduced two key modifications: (1) widening the spread and (2) incorporating an ML Alpha scalar into our optimization process.

By widening the spread to 4.0 points, we significantly decreased the likelihood of adverse selection in response to rapid price movements. This change allowed us to maintain a more stable profit profile while still profiting from the overall market trend. Furthermore, by heavily weighting the ML Alpha scalar at 5.0, we introduced an additional layer of sophistication to our model, enabling it to better capture complex order flow dynamics and adapt to changing market conditions. As a result, we observed significant improvements in our performance: Optimal PnL (optimized parameters) increased from -$6,208.19 to +$9,664.89, representing a notable gain of $15,872.08.

The optimized parameters used in this analysis are:

* Risk Aversion (0.1)
* Spread Half (4.0)
* Alpha Scalar (5.0)

We believe these findings have the potential to significantly enhance the performance of our Volcker strategy on BTCUSDT, providing a more stable and profitable trading experience for our clients.

Sincerely,

[Your Name]
Head of Quantitative Research