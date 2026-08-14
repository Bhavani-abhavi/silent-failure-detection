"""Label-free performance estimation.

Deliberately isolated from `domains` and `backtest` by an import-linter
contract: an estimator that can reach the labels it is later scored against
is not a label-free estimator, and nothing at runtime would reveal the
mistake.
"""
