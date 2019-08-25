# import pandas as pd
# from tqdm import tqdm
# from pandas.util._validators import validate_bool_kwarg
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# paso imports
# paso imports
from paso.base import pasoFunction, NameToClass
from paso.base import raise_PasoError
from paso.base import _array_to_string, pasoDecorators
from paso.base import _add_dicts
#from paso.base import _dict_value2
#from loguru import logger

from loguru import logger
import sys, os.path

### NameToClass class
class EDAToClass:
    """
        map EDAer name to class method
    """

    __EDAers__ = {
        "RandomForest": RandomForestClassifier,
        "LinearRegression": LinearRegression,
    }


### EDA
class EDA(pasoFunction):
    """
    EDA a dataset

    Note:

    Warning:

    """

    @pasoDecorators.InitWrap(narg=1)
    def __init__(self, **kwargs):

        """
        Parameters:
            filepath: (string)


        Note:

        """
        super().__init__()

    def kinds(self):
        """
        Parameters:
            None

        Returns:
            List of available models names.
        """
        return listNameToClass.__EDAers__.keys())

    @pasoDecorators.TransformWrapnarg(array=False, narg=2)
    def transform(self, X, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            inplace:
            target: dependent feature which is "target" of trainer
            verbose: (boolean) (optiona) can be set. Default:True

            kind

            Returns:
                    [train DF , test DF] SPLIT FROM X
            Raises:

            Note:
        """

        # check keywords in passes argument stream
        # non-optional kw are initiated with None

        if self.target == None:
            raise_PasoError(
                "target not specified through keyword call or ontological file for EDA: {}".format(
                    self
                )
            )

        if self.kwargs == None:
            raise_PasoError(
                "EDAt_kwargs not specified for splitter: {}".format(self)
            )

        if _check_non_optional_kw(
            self.kind,
            msg="Splitter:transform target= non-optional: {}".format(self.target),
        ):
        self.y_train = X[self.target].values
        self.X_train = X[X.columns.difference([self.target])]
        self.clf = self.model  # start with base learner
        for i, EDA_name in enumerate(self.EDA_kins):
            if self.verbose:
                logger.info("EDAer: {}".format(EDA_name))

            EDA_kind__kwargs = self.EDA_kind_kwargs_list[i]
            if self.verbose:
                logger.info("EDAer:{} kwargs: {}".format(EDA_name,EDA_kind__kwargs))
            self.EDAer = NameToClass.__EDAers__[EDA_name]()
            self.EDAer.transform( self.X_train, self.y_train,**EDA_kind__kwargs)

        return result


class Half_masked_corr_heatmap(object):
    """
    Required parameter: dataframe ... the reference pandas dataframe
    Optional parameters: title ... (string) chart title
                         file  ... (string) path+filename if you want to save image

    >>> half_masked_corr_heatmap(df,
    >>>                     'CA Housing Price Data - Variable Correlations',
    >>>                 )
    """

    def __init__(self, **kwargs):
        """
        Parameters: None


        Note:

        """
        pass


    def transform(self, X, **kwargs):
        # (dataframe, title=None, file=None):
        plt.figure(figsize=(9, 9))
        sns.set(font_scale=1)

        mask = np.zeros_like(X.corr())
        mask[np.triu_indices_from(mask)] = True

        with sns.axes_style("white"):
            sns.heatmap(X.corr(), mask=mask, annot=True, cmap='coolwarm')

        if self.title: plt.title(f'\n{self.title}\n', fontsize=18)
        plt.xlabel('')  # optional in case you want an x-axis label
        plt.ylabel('')  # optional in case you want a  y-axis label
        if self.file: plt.savefig(self.file, bbox_inches='tight')
        plt.show();

        return


class Corralation_to_target(object):
    """
    Required parameters: dataframe ... the reference pandas dataframe
                         target ... (string) column name of the target variable

    Optional parameters: title ... (string) chart title


    >>> corr_to_target(df, 'price',
    >>>           'CA Housing Price Data - Corr to Price',
               './plot_blog_images/07_corr_to_price.jpg'
    >>>                 )
    """

    def __init__(self, **kwargs):
        """
        Parameters: None


        Note:

        """
        pass

    def transform(self, X, **kwargs):
        #(X, target, title=None, file=None):
        plt.figure(figsize=(4, 6))
        sns.set(font_scale=1)

        sns.heatmap(X.corr()[[target]].sort_values(target,
                                                           ascending=False)[1:],
                    annot=True,
                    cmap='coolwarm')

        if self.title: plt.title(f'\n{self.title}\n', fontsize=18)
        plt.xlabel('')  # optional in case you want an x-axis label
        plt.ylabel('')  # optional in case you want a  y-axis label
        if self.file: plt.savefig(self.file, bbox_inches='tight')
        plt.show();

        return

    class Corralation_to_target(object):
        """
            Calculate feature correlation to target. The naive concept
            is that the lower the correlation the lower impact the feature

            >>> corr_to_target(df, 'price',
        >>>           'CA Housing Price Data - Corr to Price',
                   './plot_blog_images/07_corr_to_price.jpg'
        >>>                 )
        """

        def __init__(self, **kwargs):
            """
            Parameters: None

            """
            pass

        def transform(self, X, **kwargs):
            """
            Parameters:
                X (dataframe) the reference pandas dataframe
                target (str) column name of the target variable

                title (string) (Optional) chart title
            Return:
                result (dataframe)
            """

###
