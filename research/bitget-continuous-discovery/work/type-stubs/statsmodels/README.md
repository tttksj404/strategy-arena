# Narrow statsmodels typing boundary

The reviewer inspected the installed statsmodels **0.15.0** implementation of
`statsmodels.tsa.stattools.coint` on 2026-09-10. Its signature was
`coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None)`.
The returned `CointResult` fields were `coint_t`, `pvalue`, and `critical_values`.
The scoped stub models the numerical array call used by `x_discovery_pairs.py`;
it does not claim to type the rest of statsmodels or its future versions.

These files are `.pyi` declarations only. No runtime `statsmodels` module is
provided or shadowed. Keep the runner's locked dependency version, and recheck
the installed signature and returned fields before upgrading statsmodels.
