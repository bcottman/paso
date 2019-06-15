import math

import pandas as pd
from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from numba import jit

# paso imports
from paso.base import pasoFunction, Paso
from paso.base import pasoDecorators
from loguru import logger

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
# )
session = Paso().startup("../../parameters/default-lesson.1.yaml")


@jit
def _float_range(start, stop, step):
    istop = int((stop - start) / step)
    edges = []
    for i in range(int(istop) + 1):
        edges.append(start + i * step)
    return edges


# @jit  CANT DO IT, X IS DATAFRAME
def _fixed_width_labels(X, nbins, miny, maxy):
    # preparation of fixed-width bins

    edges = _float_range(miny, maxy, (maxy - miny) / nbins)
    lf = 0 if miny == 0 else round(abs(math.log10(abs(miny))))
    loglf = 0 if maxy == 0 else math.log10(abs(maxy / nbins))
    hf = round(abs(loglf))
    if loglf > 0:
        fs = "(%2." + str(0) + "f, %2." + str(0) + "f]"
    else:
        ff = lf + 1 if (lf > hf) else hf + 1
        fs = "(%2." + str(ff) + "f, %2." + str(ff) + "f]"

    lbl = np.array([fs % (edges[i], edges[i + 1]) for i in range(len(edges) - 1)])
    return lbl


class ContinuoustoCategory(pasoFunction):
    """
    Transforms any numpy array or any pandas dataframe
    of continuous values (``int` or ``float` arrays) to category bins.

    Parameters:
        verbose: (str) default: True
            logging on/off (``verbose=False``)


    Note:
        Datasets that are used as ``train``, ``valid``, and ``test``
        shpuld have same ``miny`` and  `` maxy``, so as to have same
        range, so as the same bin widths and labels and thus the
        same categories.

        Assumes **paso** ``EliminateUnviableFeatures()`` and other data
        cleaning steps (such as removal of Null and NA values)
        has been done previous to this.

        No matter the type of the input dataset it  will be returned
        as ``DataFrame``. If you want to set the feature names, call ``toDataFrame``
        before this function.

        Fixed-width bin, only works, WITHOUT SCALING, with datasets with multiple features
        for tree-based models such as CART, random forest, xgboost, lightgbm,
        catboost,etc. Namely Deep Learning using neural nets won't work.

        **Statistical problems with linear binning.**

        Binning increases type I and type II error; (simple proof is that as number
        of bins approaches infinity then information loss approaches zero).
        In addition, changing the number of bins will alter the bin distrution shape,
        unless the distribution is uniformLY FLAT.

        **Quantile binning can only be used with a singular data set.**

        Transforming a Continuous featuree ino a Category feature based on percentiles (QUANTILES) is WRONG
        if you have a train and test data sets. Quaniles are based on the data set and will be different unless
        each data set is distribution is equal. In rhe limit there are only two bins,
        then almost no relationship can be modeled. We are essentially doing a t-test.

        **if there are nonlinear or even nonmonotonic relationships between features**

        If you need linear binning, not quantile, use
        ``quantile_bins=False`` and specify the bin width (``delta``) or  fixed bin boundaries
        of any distribution of cuts you wish with ``nbin`` = [ cut-1, cut-2...cut-n ]

        **If you want Quantile-binning.**

        Despite the above warnings, your use case may require. qantile binning.
        Quantile based binning is a faily good strategy to use for adaptive binning.
        Quantiles are specific values or cut-points which partition
        the continuous valued distribution of a feature into
        discrete contiguous bins or intervals. Thus, q-Quantiles
        partition a numeric attribute into q equal (percetage-width) partitions.

        Well-known examples of quantiles include the 2-Quantile ,median,
        divides the data distribution into two equal (percetage-width) bins, 4-Quantiles,
        ,standard quartiles, 4 equal bins (percetage-width) and 10-Quantiles,
        deciles, 10 equal width (percetage-width) bins.

        **You should maybe looking for outliers AFTER applying a Gaussian transformation.**

    """

    def __init__(self, verbose=True):
        super().__init__()

        validate_bool_kwarg(verbose, "verbose")
        self.verbose = verbose

    def _transform(self, X, nbins, miny, maxy):
        """
        Note:
            ``quantile_bin = True``
            quantile is similar to min-max scaling:  v/(maxy-miny)
            works on any any scale
            - MIN,MAX, Ignored

            ``quantile_bin = False``
            fixed-width bin, only works, WITHOUT SCALING, with datasets with multiple features
            or tree-based models such as CART, random forest, xgboost, lightgbm,
            catboost,etc. Namely Deep Learning using neural nets won't work (correctly).
            Exception being integers that are categories, example: dayofweek

        Returns:
            pandas Series type of values type category

        """
        #        import pdb; pdb.set_trace() # debugging starts here
        if self.quantile_bins:
            # MIN,MAX, Ignored
            # quantile is similar to min-max scaling:  v/(maxy-miny)
            # works on any any scale
            return pd.qcut(X, self.nbins, duplicates="drop").rename(X.name[0] + "_qbin")
        else:
            # fixed-width bin, only works, WITHOUT SCALING, with datasets with multiple features
            # for tree-based models such as CART, random forest, xgboost, lightgbm,
            # catboost,etc. Namely Deep Learning using neural nets won't work (correctly).
            # Exception being integers that are categories, example: dayof week
            #            labels = fixed_width_labels(X,nbins,self.miny,self.maxy)
            lbls = _fixed_width_labels(X, nbins, miny, maxy)
            return pd.cut(X, nbins, labels=lbls, duplicates="drop").rename(
                X.name + "_bin"
            )

    @pasoDecorators.TransformWrap
    def transform(
        self,
        Xarg,
        drop=True,
        inplace=False,
        miny=[],
        maxy=[],
        integer=True,
        floaty=True,
        quantile_bins=True,
        nbins=100,
    ):
        """
        Transforms any float, continuous integer values of  numpy array, list, tuple or
        any pandas dataframe or series feature(s) type(s) to category type(s).

        Parameters:
            X: (dataFrame,numpy array,list)

            drop: (boolean) default:True
                do not keep original features.

            inplace: (boolean) default:False, 
                replace 1st argument with resulting dataframe.
            
            integer: (boolean) default:True
                set integer=False if not continuous and not to transform into category.

            floaty: (boolean) default:True
                set floaty=False if not continuous and not to transform into category.

            stategy: (boolean) default: True, use quantile bin.
                quantile is simular to v/(maxy-miny), works on any any scale.
                False, use fixed-width bin. miny,maxy arguments are ignored.

            nbins: (integer) default: 100

                Alternately ``nbins`` can be integer for number of bins. Or it can be
                    array of quantiles, e.g. [0, .25, .5, .75, 1.]
                    or array of fixed-width bin boundaries i.e. [0., 4., 10, 100].

                Binning, also known as quantization is used for
                    transforming continuous numeric features
                    (``np.number`` type) into ``category`` type.
                    These categories group the continuous values
                    into bins. Each bin represents a range of continuous numeric values.
                    Specific strategies of binning data include fixed-width
                    (``quantile_bins=False``) and adaptive binning (``quantile_bins = True``).

        Returns:
            (DataFrame)

        Raises:
            TypeError('"ContinuoustoCategory:inplace: requires boolean type.")
        """
        validate_bool_kwarg(integer, "integer")
        self.integer = integer
        validate_bool_kwarg(floaty, "floaty")
        self.float = floaty
        # handle miny,maxy
        self.nbins = nbins
        self.quantile_bins = quantile_bins
        # handles float, continuous integer. set integer=False if not contunuous
        # any other dataframe value type left as is.
        for nth, feature in enumerate(Xarg.columns):
            if (self.float and Xarg[feature].dtype == np.float) or (
                self.integer and Xarg[feature].dtype == np.int
            ):
                if type(self.nbins) == tuple or type(self.nbins) == list:
                    nbins = self.nbins[nth]
                else:
                    nbins = self.nbins
                tminy = Xarg[feature].min() if miny == [] else miny[nth]
                tmaxy = Xarg[feature].max() if maxy == [] else maxy[nth]
                #                print('*****  ',feature)
                Z = self._transform(Xarg[feature], nbins, tminy, tmaxy)

                # import pdb; pdb.set_trace() # debugging starts here
                # drop feature, if a list and its short, then their is an error.
                # no drop for integer=False or floaty=False
                drop = self.drop  # want the hell , do per iteration, negliable cost
                if type(self.drop) == tuple or type(self.drop) == list:
                    drop = self.drop[nth]
                if drop:
                    Xarg.drop(feature, axis=1, inplace=True)
                Xarg[Z.name] = Z
                if self.verbose:
                    logger.info(
                        "ContinuoustoCategory : {} to {}".format(feature, Z.name)
                    )
            else:
                pass

        return Xarg
