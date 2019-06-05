import math
import pandas as pd
from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from numba import jit
from tqdm import tqdm

# paso imports
from paso.base import pasoFunction, pasoDecorators
from paso.base import  _Check_No_NA_F_Values


__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#

COMPONENT_DICT = {
    "Year": 100,
    "Month": 12,
    "Week": 52,
    "Day": 31,
    "Dayofweek": 5,
    "Dayofyear": 366,
    "Elapsed": 0,
    "Is_month_end": 1,
    "Is_month_start": 1,
    "Is_quarter_end": 1,
    "Is_quarter_start": 1,
    "Is_year_end": 1,
    "Is_year_start": 1,
}
COMPONENT_DICT["Elapsed"] = (
    COMPONENT_DICT["Year"] * COMPONENT_DICT["Dayofyear"] * 24 * 3600
)  # unit=seconds)

COMPONENT_LIST = COMPONENT_DICT.keys()


class toDatetimeComponents(pasoFunction):
    """
    Parameters:
        verbose: (str) default: False
            logging on/off (``verbose=False``)
    """

    def __init__(self, verbose=False, **kwargs):

        super().__init__()
        validate_bool_kwarg(verbose, "verbose")
        self.verbose = verbose

    @pasoDecorators.TransformWrap
    def transform(
        self,
        Xarg,
        inplace=False,
        drop=True,
        dt_features=[],
        prefix=True,
        components=COMPONENT_LIST,
        **kwargs
    ):
        #    import pdb; pdb.set_trace() # debugging starts here
        """
        Note:
            Successful coercion to ``datetime`` costs approximately 100x more than if
            X[[dt_features]] was already of type datetime.

            Because of cost, a possible date will **NOT** be converted to ``datetime`` type.

            Another way, using a double negative is,
            if X[[dt_features]] is not of datetime type  (such as ``object`` type)
            then there **IS NO** attempt to coerce X[[dt_features]] to ``datetime`` type is made.

            It is best if raw data field
            is read/input in as ``datetime`` rather than ``object``. Another way, is to convert
            dataframe column using

                Xarg[features] = pd.datetimre(Xarg[feature])

        Parameters:
            Xarg: (pandas dataFrame)
                If it is not a datetime64 series, it will be converted (if possible) to one with pd.to_datetime.

            inplace : (boolean) default: False
                If True, do operation inplace and return None.


            drop: (boolean) default: True
                If True then the datetime feature/column will be removed.

            dt_features: (list) default: []
                list of column(feature) names for which datetime components
                are created.

            prefix: (boolean) default: True
                If True then the feature will be the prefix of the created datetime
                component fetures. The posfix will be _<component> to create the new
                feature column <feature>_<component>.

                if False only first 3 characters of feature string eill be used to
                create the new feature name/column <featurename[0:2]>_<component>.

            components: (list) default:
                [Year', 'Month', 'Week', 'Day','Dayofweek'
                , 'Dayofyear','Elapsed','Is_month_end'
                , 'Is_month_start', 'Is_quarter_end'
                , 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

                or set ``components`` to one or compnents names in a list
                Must be components names from default list.

        Returns:
            (DataFrame)
                toDatetimeComponents transformed into datetime feature components

       Raises:
            ValueError: if any dt_features = [].

            ValueError: if any feature has NA values.
 
        """
        self.prefix = prefix
        self.dt_features = dt_features
        self.components = components
        if self.dt_features == []:
            self.dt_features = Xarg.columns

        for feature in self.dt_features:
            _Check_No_NA_F_Values(Xarg, feature)
            if np.issubdtype(Xarg[feature].dtype, np.datetime64):
                # set new component feature name
                if self.prefix:
                    fn = feature + "_"
                else:
                    fn = feature[0 : self.PREFIX_LENGTH] + "_"

                for component in self.components:
                    if component.lower() == "Elapsed".lower():
                        Xarg[fn + "Elapsed"] = (
                            Xarg[feature].astype(np.int64) // 10 ** 9
                        ).astype(np.int32)
                    else:
                        Xarg[fn + component] = getattr(
                            Xarg[feature].dt, component.lower()
                        )  # ns to seconds
                    if self.verbose:
                        logger.info(
                            "datetime feature component added: {}".format(
                                fn + component
                            )
                        )
                if drop:
                    Xarg.drop(feature, axis=1, inplace=True)
                if self.verbose:
                    logger.info("datetime feature dropped: {}".format(feature))

        return Xarg
