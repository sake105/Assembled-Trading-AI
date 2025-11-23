import pandas as pd, pathlib as pl
p = pl.Path("output/equity_curve_1d.csv")
ec = pd.read_csv(p)
print('rows=', len(ec), 'start=', ec['equity'].iloc[0], 'end=', ec['equity'].iloc[-1])
