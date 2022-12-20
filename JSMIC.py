import numpy as np
import pandas as pd
import scipy

class JSMIC:

    # Obtain the probability value of the continuous variable by bucketing, then calculate the JS divergence, and return the result
    def JS_div(self, arr1, arr2, num_bins=20):
        max0 = max(np.max(arr1), np.max(arr2))
        min0 = min(np.min(arr1), np.min(arr2))
        if max0 == min0:
            return 0
        else:
            bins = np.linspace(min0 - 1e-4, max0 - 1e-4, num=num_bins)
            PDF1 = pd.cut(arr1, bins).value_counts() / len(arr1)
            PDF2 = pd.cut(arr2, bins).value_counts() / len(arr2)
            return self.JS_divergence(PDF1.values, PDF2.values)

    # JS divergence calculation formula
    def JS_divergence(self, p, q):
        M = (p + q) / 2
        JS_VALUE = 0.5 * scipy.stats.entropy(p, M, base=2) + 0.5 * scipy.stats.entropy(q, M, base=2)
        JS_INVERSE = 1-JS_VALUE
        return JS_INVERSE






