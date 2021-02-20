#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2月 18, 2021 

@file: utils.py
@desc: progress bar
"""
import sys
import numpy as np


def log_to_message(log, precision=4):
    fmt = "{0}: {1:." + str(precision) + "f}"
    return "    ".join(fmt.format(k, v) for k, v in log.items())


class ProgressBar(object):
    """progress bar"""

    def __init__(self, n, length=40):
        # Protect against division by zero
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i / 100.0 * n) for i in range(101)])
        self.ticks.add(n - 1)
        self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, n-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i + 1) / self.nf) * self.length))
            sys.stdout.write("\r[{0}{1}] {2}%\t{3}".format(
                "=" * b, " " * (self.length - b), int(100 * ((i + 1) / self.nf)), message
            ))
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n - 1)
        sys.stdout.write("{0}\n\n".format(message))
        sys.stdout.flush()
