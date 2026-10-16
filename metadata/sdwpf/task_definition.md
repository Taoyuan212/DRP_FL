# Forecasting task definition V1

## Frozen primary estimand

For each of 134 turbine clients, estimate quantiles q={0.1,...,0.9} of a single future active-power observation under a normal-operation target mask, conditional only on locally observed information through issue time. The primary direct horizon is 60 minutes; secondary direct horizons are 10 and 180 minutes. The history length is 6 hours.

Task B is the main experiment because official unknown/missing/abnormal targets are explicitly described as unsuitable for model evaluation. Task A (realized operational power including state outcomes) would require predicting exogenous scheduling/maintenance states that are not identified in this dataset; it remains an extension, not the main estimand.

Unknown-power observations may appear in history because they are known at issue time, but they are never scored as future labels. This avoids discarding most otherwise observable histories while preserving the official target mask. The target uses per-unit power (`Patv / 1500 kW`); processed files retain kW as well.

Main-134 is frozen for the primary experiment. Sensitivity-133 excludes client 83 only after all primary results are produced.

## Evidence adequacy

The selected 6-hour history is supported by the empirical audit in `table_lookback_diagnostics.csv`: the cross-client median Patv ACF is materially higher at 6 hours than at 12 or 24 hours, while PACF is near zero at all three long lags. Six hours is therefore the least costly defensible V1 context, not a proven optimum; 12 and 24 hours remain predeclared sensitivities. No model was trained in Stage 2.
