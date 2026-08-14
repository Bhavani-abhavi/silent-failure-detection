"""Retrospective evaluation: detection latency and false-positive rate.

Separate from `estimation` on purpose. This package holds the scoring; that
one holds the thing being scored. Folding them together makes it too easy to
tune an estimator against the number that is supposed to validate it.
"""
