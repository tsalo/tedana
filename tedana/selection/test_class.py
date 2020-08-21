"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging
import numpy as np
from scipy import stats

from tedana.stats import getfbounds
from tedana.selection._utils import getelbow, clean_dataframe

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


class DecisionTree():

    def __init__(self, function):
        self.tree = function
