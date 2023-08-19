"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np


class Constants:
    # config dict experiment_settings
    EXPERIMENT_SETTINGS = "experiment_settings"
    EXPERIMENT_SETTINGS_TYPE = "type"
    EXPERIMENT_SETTINGS_TREATMENT_UNIT_IDS = "treatment_unit_ids"
    EXPERIMENT_SETTINGS_MATCH_UNIT_SIZE = "match_unit_size"
    EXPERIMENT_SETTINGS_MATCHING_METHOD = "matching_method"
    EXPERIMENT_SETTINGS_MATCHING_START_DATE = "matching_start_date"
    EXPERIMENT_SETTINGS_MATCHING_END_DATE = "matching_end_date"
    EXPERIMENT_SETTINGS_EXPERIMENT_START_DATE = "experiment_start_date"
    EXPERIMENT_SETTINGS_EXPERIMENT_END_DATE = "experiment_end_date"
    EXPERIMENT_SETTINGS_EXCLUDE_UNIT_IDS = "exclude_unit_ids"
    EXPERIMENT_SETTINGS_MATCHING_COLUMNS = "matching_columns"
    EXPERIMENT_SETTINGS_MATCHING_WEIGHTS = "matching_weights"
    EXPERIMENT_SETTINGS_IS_CHECK_IMBALANCE = "is_check_imbalance"
    EXPERIMENT_SETTINGS_CHECK_IMBALANCE_METHOD = "check_imbalance_method"
    EXPERIMENT_SETTINGS_IS_CHECK_FLICKERS = "is_check_flickers"
    EXPERIMENT_SETTINGS_IS_REMOVE_FLICKERS = "is_remove_flickers"
    EXPERIMENT_SETTINGS_IS_CHECK_METRIC_TYPE = "is_check_metric_type"
    EXPERIMENT_SMALL_SAMPLE_ADJUSTMENT = "small_sample_adjustment"
    EXPERIMENT_SETTINGS_BOOTSTRAP_SIZE = "bootstrap_size"
    EXPERIMENT_SETTINGS_BOOTSTRAP_ITERATION = "bootstrap_iteration"
    EXPERIMENT_SETTINGS_FIXED_EFFECT_ESTIMATOR = "fixed_effect_estimator"
    EXPERIMENT_SETTINGS_INTERACTION = "interaction"
    EXPERIMENT_SETTINGS_TARGET_SAMPLE_SIZE = "target_sample_size"
    EXPERIMENT_SETTINGS_CURRENT_SAMPLE_SIZE = "current_sample_size"
    EXPERIMENT_SETTINGS_INFORMATION_RATES = "information_rates"

    # config dict columns
    COLUMNS = "columns"
    COLUMNS_COLUMN_TYPE = "column_type"
    COLUMNS_METRIC_TYPE = "metric_type"
    COLUMNS_METRIC_AGGREGATE_FUNC = "metric_aggregate_func"
    COLUMNS_METRIC_QUANTILE = "quantile"
    COLUMNS_METRIC_LOG_TRANSFORM = "log_transform"
    COLUMNS_METRIC_REMOVE_OUTLIER = "remove_outlier"
    COLUMNS_METRIC_CHECK_DISTRIBUTION = "check_distribution"
    COLUMNS_METRIC_FITTER_TYPE = "fitter_type"
    COLUMNS_METRIC_SEQUENTIAL_TESTING_TAU = "sequential_testing_tau"
    COLUMNS_CONTROL_LABEL = "control_label"
    COLUMNS_METRIC_ALPHA = "alpha"
    APPLIED_METRICS = "applied_metrics"
    APPLIED_COVARIATES = "applied_covariates"
    COLUMNS_COVARIATE_VALUE_TYPE = "value_type"
    COLUMNS_COVARIATE_NUMERATOR_COLUMN = "numerator_column"
    COLUMNS_COVARIATE_DENOMINATOR_COLUMN = "denominator_column"
    COLUMNS_DATE_FORMAT = "date_format"
    COLUMNS_EXPERIMENT_GROUP_VARIATION = "variations"
    COLUMNS_EXPERIMENT_GROUP_VARIATION_SPLIT = "variations_split"
    COLUMNS_EXPERIMENT_GROUP_CONTROL_LABEL = "control_label"
    COLUMNS_EXPERIMENT_GROUP_VARIATION_DEFAULT = ["control", "treatment"]
    COLUMNS_EXPERIMENT_GROUP_VARIATION_SPLIT_DEFAULT = [0.5, 0.5]
    COLUMNS_EXPERIMENT_GROUP_CONTROL_LABEL_DEFAULT = "control"
    COLUMNS_RATIO_METRIC_NUMERATOR_COLUMN = "numerator_column"
    COLUMNS_RATIO_METRIC_DENOMINATOR_COLUMN = "denominator_column"

    # config class default values
    EXPERIMENTVARIATION_DEFAULT_CONTROL_NAME = "control"
    EXPERIMENTVARIATION_DEFAULT_TREATMENT_NAME = "treatment"
    EXPERIMENTVARIATION_DEFAULT_CONTROL_SPLIT = 0.5
    EPXERIMENTVARIATION_DEFAULT_TREATMENT_SPLIT = 0.5

    # simulation settings
    SIMULATION_SETTINGS = "simulation_settings"
    SIMULATION_SETTINGS_ITERATION = "iteration"
    SIMULATION_SETTINGS_METRICS = "metrics"
    SIMULATION_SETTINGS_TREATMENT_EFFECT_MEAN = "treatment_effect_mean"
    SIMULATION_SETTINGS_TREATMENT_EFFECT_STD = "treatment_effect_std"
    SIMULATION_VERBOSE_LEVEL = "verbose"
    SIMULATION_EXPERIMENT_GROUP = "experiment_group"
    SIMULATION_DEFAULT_VARIATION_NAMES = [0, 1]

    # others config
    WEIGHTED_SUM_COLUMN_NAME = "weighted_sum"
    IMBALANCE_TOLERANCE = 0.1
    USE_ITERATIVE_CV_METHOD = "use_iterative_cv_method"

    # return result
    COEFFICIENT = "coefficient"
    AVERAGE_TREATMENT_EFFECT = "average_treatment_effect"
    P_VALUE = "p_value"
    CONFIDENCE_INTERVAL = "confidence_interval"
    CONFIDENCE_INTERVAL_LEFT = "confidence_interval_left"
    CONFIDENCE_INTERVAL_RIGHT = "confidence_interval_right"
    METRIC_NAME = "metric_name"
    COVARIATE_NAME = "covariate_name"
    EXPERIMENT_GROUP_NAME = "experiment_group_name"
    SE = "SE"
    JSON_MATCHING_ID = "id"
    JSON_MATCHING_SCORE = "score"
    JSON_MATCHING_RESULT = "matching_results"
    JSON_MATCHING_METHOD = "matching_method"
    JSON_MATCHING_COLUMN_NAME = "matching_column_name"
    LOG_MESSAGES = "log_messages"
    SEQUENTIAL_P_VALUE = "sequential_p_value"
    SEQUENTIAL_CONFIDENCE_INTERVAL = "sequential_confidence_interval"
    SEQUENTIAL_CONFIDENCE_INTERVAL_LEFT = "sequential_confidence_interval_left"
    SEQUENTIAL_CONFIDENCE_INTERVAL_RIGHT = "sequential_confidence_interval_right"
    SEQUENTIAL_RESULT_TYPE = "sequential_result_type"

    # additional json return result
    METRIC_VALUE = "value"
    RELATIVE_CONFIDENCE_INTERVAL = "rel_confidence_interval"
    ABSOLUTE_CONFIDENCE_INTERVAL = "abs_confidence_interval"
    SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL = "sequential_rel_confidence_interval"
    SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL = "sequential_abs_confidence_interval"
    SAMPLE_SIZE = "sample_size"
    DATA_SIZE = "data_size"
    RELATIVE_AVERAGE_TREATMENT_EFFECT = "relative_average_treatment_effect"
    AGG_FUNC_NAME = "agg_func_name"
    METRIC_RESULTS = "metric_results"
    COVARIATE_RESULTS = "covariate_results"
    EXPERIMENT_GROUP_RESULTS = "experiment_group_results"
    AGG_FUNC_RESULTS = "agg_func_results"
    TREATMENT_RESULTS = "treatment_results"
    CONTROL_RESULTS = "control_results"
    VARIATION = "variation_name"
    PREPROCESS_RESULTS = "preprocess_results"
    DOES_FLICKER_EXISTS = "does_flicker_exists"
    ARE_BUCKETS_IMBALANCED = "are_buckets_imbalanced"
    MESSAGE_COLLECTION = "message_collection"

    # diff in diff calculation
    DIFF_IN_DIFF_TREATMENT = "treatment"
    DIFF_IN_DIFF_CONTROL = "control"
    DIFF_IN_DIFF_TREATMENT_VALUE = 1
    DIFF_IN_DIFF_CONTROL_VALUE = 0
    DIFF_IN_DIFF_TIME = "time"
    DIFF_IN_DIFF_TIME_BEFORE = 0
    DIFF_IN_DIFF_TIME_AFTER = 1
    DIFF_IN_DIFF_TREATMENT_TIME_INTERACTION = "{}:{}".format(DIFF_IN_DIFF_TREATMENT, DIFF_IN_DIFF_TIME)
    DIFF_IN_DIFF_MATCHING_AGGREGATE_FUNC = np.mean

    # diff in diff plotting
    DID_PLOT_DEFAULT_X_LABEL = "Date"
    DID_PLOT_DEFAULT_Y_LABEL = "metric"
    DID_PLOT_EXPERIMENT_START_LABEL = "Experiment Start"
    DID_PLOT_DEFAULT_TREATMENT_EFFECT_PLOT_TITLE = "Diff-in-Diff treatment effect"
    DID_PLOT_DEFAULT_MATCHING_PLOT_TITLE = "Diff-in-Diff matching"
    DID_PLOT_DEFAULT_FIGURE_SIZE = (20, 7)

    # bootstrap
    BOOTSTRAP_SE = "bootstrap_se"
    BOOTSTRAP_T = "bootstrap_t"

    # fitter
    STATISTICS_CALCULATE_FUNC_QUANTILE = "quantile"
    STATISTICS_CALCULATE_FUNC_CONTROL_LABEL = "control_label"
    STATISTICS_CALCULATE_FUNC_TREATMENT_LABEL = "treatment_label"
    STATISTICS_CALCULATE_FUNC_METRIC = "metric"
    STATISTICS_CALCULATE_FUNC_CLUSTER = "cluster"
    STATISTICS_CALCULATE_FUNC_EXPERIMENT_GROUP = "experiment_group"
    STATISTICS_CALCULATE_FUNC_NUMERATOR_COLUMN = "numerator_column"
    STATISTICS_CALCULATE_FUNC_DENOMINATOR_COLUMN = "denominator_column"
    FIXED_EFFECT_DEMEAN_COLUMN_POSTFIX = "fixed_effect_demean"
    ALL_DATA_GROUP = "all_data"
    DEFAULT_ALPHA = 0.05

    # power calculator
    POWER = "power"
    ABS_TREATMENT_EFFECT = "absolute_treatment_effect"
    SAMPLE_SIZE_PER_GROUP = "sample_size_per_group"
    OUTPUT_METRIC = "metric"
    POWER_CALCULATOR_STANDARD_DEVIATION = "standard_deviation"
    POWER_CALCULATOR_METRIC_VALUE = "metric_value"
    POWER_CALCULATOR_SAMPLE_SIZE = "sample_size"
    POWER_CALCULATOR_DATA_SIZE = "data_size"

    # logging
    INFO = "info"
    WARNINGS = "warnings"
    ERRORS = "errors"

    # data loader storage types
    PANDAS_FORMAT = "pandas"
    PARQUET_FORMAT = "parquet"

    # SRM
    IMBALANCE_BINOMIAL_THRESHOLD = 0.01
    IMBALANCE_CHI_SQUARE_THRESHOLD = 0.01
    SRM_P_VALUE_THRESHOLD = 0.01
    SRM_CARDINALITY_THRESHOLD = 50
    SRM_MIN_OBSERVATION_COUNT = 10

    # distribution
    DISTRIBUTION_SAMPLE_SIZE_THRESHOLD = 8
    DISTRIBUTION_SKEW_PVALUE_THRESHOLD = 0.05
