"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_platform.src.models.configuration_model.config import DiDConfig
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.diff_in_diff.prep_data import (
    get_aggregate_metric_in_unit_ids,
    get_data_between_start_end_date,
)


def prep_matching_plot_data(
    data: pd.DataFrame,
    config: DiDConfig,
    control_unit_ids: List[int],
    y_metric: str = Constants.WEIGHTED_SUM_COLUMN_NAME,
    matching_aggregate_func: Callable = Constants.DIFF_IN_DIFF_MATCHING_AGGREGATE_FUNC,
) -> Tuple[pd.Series, pd.Series]:
    """function to prepare data for plot_matching_parallel_lines. This function split pre-experiment
        data into treatment and control, aggregate data on daily level.

    Arguments:
        data {pd.DataFrame} -- data
        config {DiDConfig} -- diff-in-diff config object
        control_unit_ids {List[int]} -- list of control unit ids
        y_metric {str} -- the metric to plot parallel on (the y-axis of the plot)
        matching_aggregate_func {Callable} -- how we aggregate metric across units (i.e. sum)

    Returns:
        Tuple[pd.Series, pd.Series] -- [description]
    """
    matching_data = get_data_between_start_end_date(
        data,
        config.date.column_name,
        config.matching_start_date,
        config.matching_end_date,
    )

    matching_control_aggregate_metric_series = get_aggregate_metric_in_unit_ids(
        matching_data,
        config.date.column_name,
        config.experiment_randomize_units[0].column_name,
        control_unit_ids,
        y_metric,
        matching_aggregate_func,
    )

    matching_treatment_aggregate_metric_series = get_aggregate_metric_in_unit_ids(
        matching_data,
        config.date.column_name,
        config.experiment_randomize_units[0].column_name,
        config.treatment_unit_ids,
        y_metric,
        matching_aggregate_func,
    )

    return (
        matching_treatment_aggregate_metric_series,
        matching_control_aggregate_metric_series,
    )


def prep_treatment_effect_plot_data(
    data: pd.DataFrame,
    config: DiDConfig,
    y_metric: str,
    matching_aggregate_func: Callable,
) -> pd.DataFrame:
    """function to prepare data for plot_treatment_effect()

    Arguments:
        data {pd.DataFrame} -- should be data output from prep_data_for_diff_in_diff()
        config {DiDConfig} -- config object

    Keyword Arguments:
        y_metric {str} -- the metric to plot on (the y-axis of the plot)
        matching_aggregate_func {Callable} -- how we aggregate metric across units (i.e. sum)

    Returns:
        [pd.DataFrame] -- dataframe with date as index. There are two columns: treatment & control
    """
    plot_data = (
        data.groupby([config.date.column_name, Constants.DIFF_IN_DIFF_TREATMENT])[y_metric]
        .apply(matching_aggregate_func)
        .unstack()
    )
    return plot_data


def plot_matching_parallel_lines(
    treatment_units_aggregate_metric_series: pd.Series,
    control_units_aggregate_metric_series: pd.Series,
    control_unit_ids: List[int],
    treatment_unit_ids: List[int],
    figsize: Tuple = Constants.DID_PLOT_DEFAULT_FIGURE_SIZE,
    title: str = Constants.DID_PLOT_DEFAULT_MATCHING_PLOT_TITLE,
    xlabel: str = Constants.DID_PLOT_DEFAULT_X_LABEL,
    ylabel: str = Constants.DID_PLOT_DEFAULT_Y_LABEL,
):
    """Function to plot matching quality(parallelism between treatment and control).
        This function generates a plot where x-axis is date, y-axis is the chosen metric,
        legend is treatment group and control group.

    Arguments:
        config {DiDConfig} -- config
        treatment_units_aggregate_metric_series {pd.Series} -- treatment value with index as date
        control_units_aggregate_metric_series {pd.Series} -- control value with index as date
        control_unit_ids {List[int]} -- List of control unit ids
        treatment_unit_ids {List[int]} -- List of treatment unit ids

    Keyword Arguments:
        figsize {Tuple} -- the size of the figure (default: {(20, 7)})
        title {Optional[str]} -- title of the plot (default: {None})
        xlabel {Optional[str]} -- x label of the plot (default: {None})
        ylabel {Optional[str]} -- y label of the plot (default: {None})
    """

    ax = treatment_units_aggregate_metric_series.plot(legend=True, figsize=figsize, color=["orange"])
    control_units_aggregate_metric_series.plot(legend=True, ax=ax, color=["cornflowerblue"])

    if len(treatment_unit_ids) <= 15 and len(control_unit_ids) <= 15:
        plt.legend(
            [
                "{}: {}".format(Constants.DIFF_IN_DIFF_TREATMENT, treatment_unit_ids),
                "{}: {}".format(Constants.DIFF_IN_DIFF_CONTROL, control_unit_ids),
            ]
        )
    else:
        plt.legend([Constants.DIFF_IN_DIFF_TREATMENT, Constants.DIFF_IN_DIFF_CONTROL])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_diff_in_diff_treatment_effect(
    data: pd.DataFrame,
    config: DiDConfig,
    figsize: Tuple[int, int] = Constants.DID_PLOT_DEFAULT_FIGURE_SIZE,
    title: str = Constants.DID_PLOT_DEFAULT_TREATMENT_EFFECT_PLOT_TITLE,
    xlabel: str = Constants.DID_PLOT_DEFAULT_X_LABEL,
    ylabel: str = Constants.DID_PLOT_DEFAULT_Y_LABEL,
) -> None:
    ax = data.plot(figsize=figsize, color=["cornflowerblue", "orange"])
    ax.axvline(x=[np.datetime64(config.experiment_start_date)], color="red", linestyle="--")
    ax.legend([Constants.DIFF_IN_DIFF_CONTROL, Constants.DIFF_IN_DIFF_TREATMENT])

    # add a line to the plot to indicate experiment start
    ax.text(
        x=np.datetime64(config.experiment_start_date),
        y=int(np.median(data)),
        s=Constants.DID_PLOT_EXPERIMENT_START_LABEL,
        horizontalalignment="left",
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
