# import pandas as pd
# from tqdm import tqdm
# from pandas.util._validators import validate_bool_kwarg
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# paso imports
from paso.base import pasoFunction, NameToClass
from paso.base import raise_PasoError
from paso.base import _array_to_string, pasoDecorators

from loguru import logger


### NameToClass class
class EDAToClass:
    """
        map EDAer name to class method
    """

    __EDAers__ = {
        "Half_masked_corr_heatmap": Half_masked_corr_heatmap,
        "Corralation_to_target": Corralation_to_target,
        'Feature_to_targets_scatterplots': Feature_to_targets_scatterplots,
    }


### EDA
class EDA(pasoFunction):
    """
    EDA a dataset

    Note:

    Warning:

    """

    @pasoDecorators.InitWrap()
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
        return listNameToClass.__EDAers__.keys()

    @pasoDecorators.TransformWrapnarg(array=False)
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
                "EDA_kwargs not specified for kind: {}".format(self)
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

### Half_masked_corr_heatmap
class Half_masked_corr_heatmap(object):
    """
    Required parameter: X ... the reference pandas X
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
        # (X, title=None, file=None):
        rcParams["figure.figsize"] = (self.x_size, self.y_size)
        sns.set(font_scale=self.font_scale)

        mask = np.zeros_like(X.corr())
        mask[np.triu_indices_from(mask)] = True
        result = X.corr()
        with sns.axes_style("white"):
            sns.heatmap(result, mask=mask, annot=True, cmap=self.cmap)

        if self.title: plt.title(f'\n{self.title}\n', fontsize=18)
        plt.xlabel('')  # optional in case you want an x-axis label
        plt.ylabel('')  # optional in case you want a  y-axis label
        if self.file: plt.savefig(self.file, bbox_inches='tight')
        plt.show();

        return result

### Corralation_to_target
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
            X (X) the reference pandas X
            target (str) column name of the target variable

            title (string) (Optional) chart title
        Return:
            result (X)
        """
    #(X, target, title=None, file=None):
    plt.figure(figsize=(4, 6))
    sns.set(font_scale=1)

    result = X.corr()[[target]].sort_values(target,ascending=False)[1:]
    sns.heatmap(resultannot=True, cmap='coolwarm')

    if self.title: plt.title(f'\n{self.title}\n', fontsize=18)
    plt.xlabel('')  # optional in case you want an x-axis label
    plt.ylabel('')  # optional in case you want a  y-axis label
    if self.file: plt.savefig(self.file, bbox_inches='tight')
    plt.show();

    return result

### Feature_to_targets_catterplots
class Feature_to_targets_scatterplots(object):
    """
        from Stackoverflow, Alexander McFarlane
        https://stackoverflow.com/questions/7066121/
        Plot feature  to target scatterplots.

    """

    def __init__(self, **kwargs):
        """
        Parameters: None

        """
        pass

    def transform(self, X, **kwargs):
        """
        Parameters:
            X (X) the reference pandas X
            target (str) column name of the target variable
            title (string) (Optional) chart title
            file (string) (Optional) path+filename if you want to save image

        Returns: self

        """

#        def gen_scatterplots(X, target_column, list_of_columns, cols=1, file=None):
            rows = math.ceil(len(list_of_columns) / cols)
            figwidth = 5 * cols
            figheight = 4 * rows

            fig, ax = plt.subplots(nrows=rows,
                                   ncols=cols,
                                   figsize=(figwidth, figheight))

            color_choices = ['blue', 'grey', 'goldenrod', 'r', 'black', 'darkorange', 'g']

            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            ax = ax.ravel()  # Ravel turns a matrix into a vector... easier to iterate

            for i, column in enumerate(list_of_columns):
                ax[i].scatter(X[column],
                              X[target_column],
                              color=color_choices[i % len(color_choices)],
                              alpha=0.1)


                ax[i].set_ylabel(f'{target_column}', fontsize=14)
                ax[i].set_xlabel(f'{column}', fontsize=14)

            fig.suptitle('\nEach Feature vs. Target Scatter Plots', size=24)
            fig.tight_layout()
            fig.subplots_adjust(bottom=0, top=0.88)
            if file: plt.savefig(file, bbox_inches='tight')
            plt.show();

            return self


def plot_confusion_matrix(self, cm, normalize=False):
    # Plot non-normalized confusion matrix
    if normalize:
        title = (
                "Confusion matrix, with normalization"
                + self.model_name
                + "  "
                + self.model_type
        )
    else:
        title = "Confusion matrix " + self.model_name + "  " + self.model_type

    self._plot_confusion_matrix(
        cm, self.class_names, normalize=normalize, title=title
    )

    plt.show()


def _plot_confusion_matrix(
        self, cm, class_names, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function gprahically plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        if self.verbose:
            logger.info("Normalized confusion matrix")
    else:
        if self.verbose:
            logger.info("Confusion matrix, without normalization")

    fig, ax = plt.subplots()

    #        rcParams["figure.figsize"] = (self.x_size, self.y_size)
    rcParams["figure.figsize"] = (6.0, 6.0)
    #        np.set_printoptions(precision=self.precision)
    np.set_printoptions(precision=3)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".3f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax

