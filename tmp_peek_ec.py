import pandas as pd
ec = pd.read_csv(r"output/equity_curve_1d.csv")
print("Rows:", len(ec), "Start:", ec["equity"].iloc[0], "End:", ec["equity"].iloc[-1])
print(ec.tail(5))
