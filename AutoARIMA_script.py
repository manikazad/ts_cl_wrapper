
__author__ = 'Actify Data Labs'

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
import argparse
from numpy import exp, log
import numpy as np
from numpy import polyfit
from scipy.stats import boxcox
import pmdarima as pm
import pickle
from functools import partial
from dateparser import parse as parse_date



def adfuller_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    result_dict = {}
    result_dict['adf_statistic'] = result[0]
    result_dict['p_value'] = result[1]
    result_dict['used_lag'] = result[2]
    result_dict['n_obs'] = result[3]
    result_dict['critical_values'] = result[4]

    return result_dict


def forecast_accuracy(forecast, actual, err_metric='all'):

    if err_metric == 'all':
        mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
        me = np.mean(forecast - actual)  # ME
        mae = np.mean(np.abs(forecast - actual))  # MAE
        mpe = np.mean((forecast - actual) / actual)  # MPE
        rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
        corr = np.corrcoef(forecast, actual)[0, 1]  # corr
        mins = np.amin(np.hstack([forecast[:, None],
                                  actual[:, None]]), axis=1)
        maxs = np.amax(np.hstack([forecast[:, None],
                                  actual[:, None]]), axis=1)
        minmax = 1 - np.mean(mins / maxs)  # minmax
        acf1 = acf(forecast - actual)[1]  # ACF1
        return ({'mape': mape, 'me': me, 'mae': mae,
                 'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
                 'corr': corr, 'minmax': minmax})

    elif err_metric == 'mape':
        mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
        return mape

    elif err_metric == 'me':
        me = np.mean(forecast - actual)  # ME
        return me

    elif err_metric == 'mae':
        mae = np.mean(np.abs(forecast - actual))  # MAE
        return mae

    elif err_metric == 'mpe':
        mpe = np.mean((forecast - actual) / actual)  # MPE
        return mpe

    elif err_metric == 'rmse':
        rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
        return rmse

    elif err_metric == 'corr':
        corr = np.corrcoef(forecast, actual)[0, 1]  # corr
        return corr

    elif err_metric == 'mins':
        mins = np.amin(np.hstack([forecast[:, None],
                                  actual[:, None]]), axis=1)
        return mins

    elif err_metric == 'maxs':
        maxs = np.amax(np.hstack([forecast[:, None],
                                  actual[:, None]]), axis=1)
        return maxs

    elif err_metric == 'minmax':
        mins = np.amin(np.hstack([forecast[:, None],
                                  actual[:, None]]), axis=1)
        maxs = np.amax(np.hstack([forecast[:, None],
                                  actual[:, None]]), axis=1)
        minmax = 1 - np.mean(mins / maxs)  # minmax
        return minmax

    elif err_metric == 'acf1':
        acf1 = acf(forecast - actual)[1]  # ACF1
        return acf1


def power_transform(data):
    transformed, lmbda = boxcox(data)
    return transformed, lmbda


def invert_boxcox(value, lam):
    # log case
    if lam == 0:
        return exp(value)
    # all other cases
    return exp(log(lam * value + 1) / lam)


def inverse_power_transform(series, lmbda):
    return pd.Series([invert_boxcox(x, lmbda) for x in series])


def seasonal_trend_removal(series, date_col, data_freq, seasonality, poly_degree=4):
    """
    Params:
    series: time series data, along with date/time column
    data_freq: frequency of the data, options are 'D' (daily), 'W' (weekly), 'M'(monthly), 'Q'(quaterly),
    'H' (half yearly), 'Y'(Yearly).
    seasonalilty: expected seasonality of the data, options are 'D' (daily), 'W' (weekly), 'M'(monthly), 'Q'(quaterly),
    'H' (half yearly), 'Y'(Yearly).
    """

    freq_dict = {
        'D': {'W': 7, 'M': 30, 'Q': 121, 'H': 182, 'Y': 365},
        'W': {'M': 4, 'H': 21, 'Y': 52},
        'M': {'Q': 3, 'H': 6, 'Y': 12}
    }
    n = freq_dict[data_freq][seasonality]

    X = [i % n for i in range(0, len(series))]
    y = series.values

    coef = polyfit(X, y, poly_degree)

    ## Generated curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(poly_degree):
            value += X[i] ** (poly_degree - d) * coef[d]
        curve.append(value)

    ## Take seasonal fit difference
    values = series.values
    diff = list()
    for i in range(len(values)):
        value = values[i] - curve[i]
        diff.append(value)

    return diff, curve


def get_data_freq(series, date_col):
    date_diff = pd.to_datetime(series[date_col]).diff()


def trend_diff(series, period=1):
    return series.diff()


def invert_trend_trans(tran_series, original, period):
    tran_series[0:period] = original[0:period]
    return tran_series.cumsum()


def get_prediction(model, period):
    pass


def main():

    parser = argparse.ArgumentParser()

    # Method Argument
    parser.add_argument("--verbose", action='store_true', help="verbose : bool, optional, default False",
                        default=False)

    parser.add_argument("--run_test", action='store_true', default=False, help="Runs test")

    parser.add_argument("--training",
                        help='Boolean argument for running training',
                        default=False,
                        action='store_true',)
    parser.add_argument("--testing",
                        help='Boolean argument for running testing',
                        default=False,
                        action='store_true')
    parser.add_argument("--train_test", action='store_true', help='Boolean argument for running training and '
                                                                  'testing simultaneously', default=False)
    parser.add_argument("--predict", action='store_true',
                        help='Boolean argument for running the predictions on a '
                             'pretrained model. Model input file is neccessary, Returns predicted values. and stores '
                             'them in result file', default=False)

    parser.add_argument("--result_path", action='store', help="Result output file path.")
    parser.add_argument("--input_model", action='store', help="Input Model Address")
    parser.add_argument("--model_output_path", action='store', help="Output Model File Path")

    # Data Arguments
    parser.add_argument("--series",
                        help="Time series data file (csv only) path, Data should have a date column. ",
                        action='store')
    parser.add_argument("-S", "--train_test_split",
                        help="Fraction of data to be used for training the model "
                             "required if train test data is not pre split",
                        type=float,
                        default=0.8,
                        action='store')

    parser.add_argument("--train_series",
                        help="training time series data file (csv only) path ",
                        action='store')
    parser.add_argument("--test_series",
                        help="testing time series data file (csv only) path ",
                        action='store')
    parser.add_argument("--pre_split",
                        help="Boolean Indicator variable, which tells if the train test data is splitted beforehand",
                        default=False,
                        action='store_true')

    parser.add_argument("--n_predict_periods",
                        help="number of future time periods for which prediction should be made",
                        type=int,
                        default=1,
                        action='store')

    parser.add_argument("--date_index",
                        help="Column name containing dates, which will be used as the data index",
                        type=str,
                        action='store')
    parser.add_argument("--date_format",
                        help="Give the date format in data",
                        type=str,
                        default="%Y-%m-%d",
                        action='store')

    parser.add_argument("--endog",
                        help="Column name containing endogenous variable, "
                             "If not given then the first column of the csv will be considered the endogenous variable "
                             "If exogs are given then the remaining column is considered as endog",
                        type=str,
                        action='store')
    parser.add_argument("--exog",
                        help="Optional, Column name containing exogenous variable",
                        type=str,
                        nargs='*',
                        required=False,
                        action='store')

    # Model Parameter Arguments
    parser.add_argument("--error_metric",
                        help="String, Default 'all', Error metric to be used. Available options are "
                             "'mse'(mean squared error), "
                             "'mae' (mean absolute error), "
                             "'rmse', (root mean square error) "
                             "'me' (mean error), "
                             "'mpe' (mean percentage error), "
                             "'corr' (error correlation coeficient),"
                             "'mins', 'maxs', 'minmax', 'acf1', 'all' (Calculates all of the above)",
                        default='all',
                        action='store')

    parser.add_argument("--freq",
                        help="str, optional"
                             "The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',"
                             "'M', 'A', or 'Q'. This is optional if dates are given.",
                        type=str,
                        action='store')

    parser.add_argument("--seasonal",
                        help="bool, optional, (default False)",
                        default=False,
                        action='store_true')
    parser.add_argument("--m",
                        help="int, optional (default=1) "
                             "The period for seasonal differencing, 'm' refers to the number of "
                             "periods in each season. For example, 'm' is 4 for quarterly data, 12 "
                             "for monthly data, or 1 for annual (non-seasonal) data. Default is 1. "
                             "Note that if 'm' == 1 (i.e., is non-seasonal), ''seasonal'' will be "
                             "set to False. ",
                        type=int,
                        default=1,
                        action='store')

    # parser.add_argument("--poly_fit_season",
    #                     help="bool, optional, (default False)"
    #                          "If seasonal is True and this option is chosen then a polynomial curve is fit over "
    #                          "training data to estimate the seasonal component. This works better ang gives smoother "
    #                          "predictions than differencing as the seasonal component is estimated over multiple "
    #                          "cycle rather than just differencing one cycle from the whole series.",
    #                     default=False,
    #                     action='store_true')
    # parser.add_argument("--no_power_transform",
    #                     help="bool, optional, (default True)"
    #                          "Removes any exponential growth trend in the time series."
    #                          "Obtained series is more likely to be stationary with lesser number of trend "
    #                          "differencing. Detects the degree of power transform to carry out using Box-Cox,",
    #                     default=True,
    #                     action='store_false')

    parser.add_argument("--stationary",
                        help="bool, optional, (default False)",
                        default=False,
                        action='store_true')
    parser.add_argument("--trend",
                        help="str {'c','nc'}, (default None) "
                             "Whether to include a constant or not. 'c' includes constant, "
                             "'nc' no constant.",
                        type=str,
                        default=None,
                        action='store')

    parser.add_argument("--d",
                        help="int, optional, (default None) "
                             "The order of first-differencing. "
                             "If None (by default), the value will automatically be selected based on the "
                             # "results of the 'test' (i.e., either the Kwiatkowski Phillips Schmidt Shin, "
                             "Augmented Dickey-Fuller or the Phillips Perron test will be conducted to find "
                             "the most probable value). Must be a positive integer or None.",
                        type=int,
                        default=None,
                        action='store')
    parser.add_argument("--start_p",
                        help="int, optional, (default 1) "
                             "The starting value of 'p', the order (or number of time lags) "
                             "of the auto-regressive ('AR') model. Must be a positive integer.",
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument("--start_q",
                        help="int, optional, (default 1) "
                             "The starting value of 'q', the order of the moving-average "
                             "('MA') model. Must be a positive integer.",
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument("--max_p",
                        help="int, optional, (default 5) "
                             "The maximum value of 'p', inclusive. Must be a positive integer "
                             "greater than or equal to 'start_p'.",
                        type=int,
                        default=5,
                        action='store')
    parser.add_argument("--max_q",
                        help="int, optional, (default 5) "
                             "The maximum value of 'q', inclusive. Must be a positive integer "
                             "greater than 'start_q'.",
                        type=int,
                        default=5,
                        action='store')
    parser.add_argument("--max_d",
                        help="int, optional, (default 2) "
                             "The maximum value of 'd', inclusive. Must be a positive integer "
                             "greater than 'd'.",
                        type=int,
                        default=2,
                        action='store')
    parser.add_argument("--D",
                        help="int, optional (default=None)"
                             "The order of the seasonal differencing. If None (by default, the value"
                             "will automatically be selected based on the results of the"
                             "'seasonal_test'. Must be a positive integer or None.",
                        type=int,
                        default=None,
                        action='store')

    parser.add_argument("--start_P",
                        help="int, optional (default=1) "
                             "The starting value of 'P', the order of the auto-regressive portion "
                             "of the seasonal model.",
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument("--start_Q",
                        help="int, optional (default=1) "
                             "The starting value of 'Q', the order of the moving-average portion "
                             "of the seasonal model.",
                        type=int,
                        default=1,
                        action='store')

    parser.add_argument("--max_P",
                        help="int, optional (default=2) "
                             "The maximum value of 'P', inclusive. Must be a positive integer "
                             "greater than 'start_P'.",
                        type=int,
                        default=2,
                        action='store')
    parser.add_argument("--max_Q",
                        help="int, optional (default=2) "
                             "The maximum value of 'Q', inclusive. Must be a positive integer "
                             "greater than 'start_Q'.",
                        type=int,
                        default=2,
                        action='store')
    parser.add_argument("--max_D",
                        help="int, optional (default=1) "
                             "The maximum value of ''D''. Must be a positive integer greater "
                             "than 'D'.",
                        type=int,
                        default=1,
                        action='store')

    parser.add_argument("--information_criterion",
                        help="str, optional (default='aic') "
                             "The information criterion used to select the best ARIMA model. "
                             "Options are  ('aic', 'bic', 'hqic', 'oob').",
                        type=str,
                        default='aic',
                        action='store')

    parser.add_argument("--alpha",
                        help="float, optional (default=0.05) "
                             "Level of the test for testing significance.",
                        type=float,
                        default=0.05,
                        action='store')
    parser.add_argument("--stationary_test",
                        help="str, optional (default='kpss') "
                             "Type of unit root test to use in order to detect stationarity if "
                             "''stationary'' is False and ''d'' is None. Default is 'kpss' "
                             "(Kwiatkowski Phillips Schmidt Shin). Other option is 'adf' (Augmented Dickey-Fuller)" ,
                        type=str,
                        dest='test',
                        default='adf',
                        action='store')
    parser.add_argument("--seasonal_test",
                        help="str, optional (default='ocsb') "
                             "This determines which seasonal unit root test is used if ''seasonal'' "
                             "is True and ''D'' is None. Default is 'OCSB'.",
                        type=str,
                        default='ocsb',
                        action='store')


    parser.add_argument("--no_transparams",
                        help="bool, optional "
                             "Whehter or not to transform the parameters to ensure stationarity. "
                             "If False, no checking for stationarity or invertibility is done.",
                        dest='transparams',
                        default=True,
                        action='store_false')

    parser.add_argument("--method",
                        help="method : str {'css-mle','mle','css'}, (default None)"
                             "This is the loglikelihood to maximize.  If 'css-mle'', the "
                             "conditional sum of squares likelihood is maximized and its values "
                             "are used as starting values for the computation of the exact "
                             "likelihood via the Kalman filter.  If 'mle', the exact likelihood "
                             "is maximized via the Kalman Filter.  If 'css' the conditional sum "
                             "of squares likelihood is maximized.  All three methods use "
                             "'start_params' as starting parameters.",
                        type=str,
                        default=None,
                        action='store')
    parser.add_argument("--solver",
                        help="str or None, optional "
                             "Solver to be used.  The default is 'lbfgs' (limited memory "
                             "Broyden-Fletcher-Goldfarb-Shanno).  Other choices are 'bfgs', "
                             "'newton' (Newton-Raphson), 'nm' (Nelder-Mead), 'cg' - "
                             "(conjugate gradient), 'ncg' (non-conjugate gradient), and "
                             "'powell'. By default, the limited memory BFGS uses m=12 to "
                             "approximate the Hessian, projected gradient tolerance of 1e-8 and "
                             "factr = 1e2. You can change these by using kwargs.",
                        type=str,
                        default='lbfgs',
                        action='store')
    parser.add_argument("--maxiter",
                        help="int, optional (default=None) "
                             "The maximum number of function evaluations. Statsmodels defaults this "
                             "value to 50 for SARIMAX models and 500 for ARIMA and ARMA models. If "
                             "passed as None, will use the seasonal order to determine which to use "
                             "(50 for seasonal, 500 otherwise).",
                        type=int,
                        default=None,
                        action='store')
    parser.add_argument("--out_of_sample_size",
                        help="int, optional (default=0) "
                             "The ''ARIMA'' class can fit only a portion of the data if specified, "
                             "in order to retain an 'out of bag' sample score. This is the number of "
                             "examples from the tail of the time series to hold out and use as "
                             "validation examples. The model will not be fit on these samples, but "
                             "the observations will be added into the model's 'endog' and 'exog' "
                             "arrays so that future forecast values originate from the end of the endogenous vector.",
                        type=int,
                        default=0,
                        action='store')
    parser.add_argument("--scoring",
                        help="str, optional (default='mse') "
                             "If performing validation (i.e., if ''out_of_sample_size'' > 0), the "
                             "metric to use for scoring the out-of-sample data. One of ('mse', 'mae')",
                        type=str,
                        default='mse',
                        action='store')
    parser.add_argument("--tol",
                        help="tol : float "
                             "The convergence tolerance.  Default is 1e-08.",
                        type=float,
                        default=1e-08,
                        action='store')
    parser.add_argument("--disp",
                        help="int, optional "
                             "If True, convergence information is printed.  For the default "
                             "l_bfgs_b solver, disp controls the frequency of the output during "
                             "the iterations. disp < 0 means no output in this case.",
                        type=int,
                        default=0,
                        action='store')


    parsed_args = parser.parse_args()

    # Checking for necessary input arguments in each case

    if parsed_args.run_test:

        parsed_args.endog = ['Passengers']
        parsed_args.date_index = ['Month']

        series = load_data('./data/airline-passengers.csv', date_column='Month', date_format="%Y-%m")
        print("Data Loaded with {} rows and {} columns".format(series.shape[0], series.shape[1]))
        assert series.shape[0] == 144, "Test failed, row counts didn't match"
        assert series.shape[1] == 1, "Test failed, column counts didn't match"

        train, test = train_test_split(series, split_perc=0.67)
        print("Data split into train ({} rows) and test ({} rows)".format(train.shape[0], test.shape[0]))
        assert train.shape[0] == 96, "Test failed, train rows counts didn't match"
        assert test.shape[0] == 48, "Test failed, test rows counts didn't match"

        params = get_params(parsed_args)
        print("Model Parameters : \n", params)

        test_score1, predictions1 = train_test(train, test, params, './models/arima_test.pkl',
                                             err_metric=parsed_args.error_metric)

        print("Model Trained")
        print("Model Tested over test data, Test score ('all') is :", test_score1)

        print("Test Successful")

        model = train_model(train, params)
        test_score2, predictions2 = test_model(model, test, params)

        assert test_score1 == test_score2, "Test Failed: All accuracy scores are not same"
        assert predictions1.all() == predictions2.all(), "Test Failed: All predictions are not same"

    elif parsed_args.training:

        assert parsed_args.train_series, "Please provide the train data path argument (--train_series)"
        assert parsed_args.date_index, "Please provide the date column argument (--date_index)"
        assert parsed_args.model_output_path, "Please provide output model path argument (--model_output_path)"

        series = load_data(parsed_args.train_series, date_column=parsed_args.date_index)
        params = get_params(parsed_args)
        model = train_model(series, params, exog=parsed_args.exog)
        dump_model(model, parsed_args.model_output_path)
        print("Dumped the model at:", parsed_args.model_output_path)


    elif parsed_args.testing:

        assert parsed_args.test_series, "Please provide the test data path argument (--train_series)"
        assert parsed_args.date_index, "Please provide the date column argument (--date_index)"
        assert parsed_args.input_model, "Please provide input model path argument (--input_model)"

        series = load_data(parsed_args.test_series, date_column=parsed_args.date_index)
        params = get_params(parsed_args)
        model = load_model(parsed_args.input_model)

        accuracy, forecasts = test_model(model, series, params, exog=parsed_args.exog, err_metric = parsed_args.error_metric)

        print("Test Score ({}): ".format(parsed_args.error_metric), accuracy)


    elif parsed_args.train_test:
        if parsed_args.pre_split:
            assert parsed_args.train_series, "Please provide a train series data"
            assert parsed_args.test_series, "Please provide a test series data"
            assert parsed_args.model_output_path, "Please provide a model output path"

            train_series = load_data(parsed_args.train_series)
            test_series = load_data(parsed_args.test_series)

        else:
            assert parsed_args.series, "Please provide series data"
            assert parsed_args.train_test_split, "Please provide split ratio between train and test data"

            series = load_data(parsed_args.series, parsed_args.date_index)
            train_series, test_series = train_test_split(series, parsed_args.train_test_split)

        params = get_params(parsed_args)

        test_score, predictions = train_test(train_series, test_series, params,
                                             parsed_args.model_output_path,
                                             err_metric=parsed_args.error_metric)

        print("The model is dumped at following location:", parsed_args.model_output_path)
        print("Test Score ({}): ".format(parsed_args.error_metric), test_score)

    elif parsed_args.predict:
        assert parsed_args.input_model, "Please provide an Input Model Path"
        assert parsed_args.n_predict_periods, "Please provide number of periods for prediction"
        assert parsed_args.result_path, "Please provide output result file path"

        model = load_model(parsed_args.input_model)
        predicted = model.predict(parsed_args.n_predict_periods,
                                  exogenous=parsed_args.exog,
                                  return_conf_int=True,
                                   alpha=parsed_args.alpha)
        pred_df = pd.DataFrame(predicted[0], columns=['predictions'])

        print("Prediction Successful.")
        print(pred_df)
        print("The Results are being dumped at following location: ", parsed_args.result_path)
        pred_df.to_csv(parsed_args.result_path)


def train_test_split(series, split_perc=0.8):
    offset = int(series.shape[0] * split_perc)

    train = series[:offset]
    test = series[offset:]

    return train, test


def dump_model(model, model_output_path):
    with open(model_output_path, 'wb') as _fp:
        pickle.dump(model, _fp)
    return "Success"


def load_model(model_path):
    with open(model_path, 'rb') as fp:
        model = pickle.load(fp)
        return model


def load_data(data_path, date_column, date_format="%Y-%m-%d"):
    # parse_dt = partial(parse_date, date_formats=[date_format]) # date_parser=parse_dt,
    return pd.read_csv(data_path, index_col=date_column, parse_dates=True, infer_datetime_format=True)


def test_model(model, test_series, params,  exog=None,  err_metric='all'):

    if exog:
        test_series = test_series[set(test_series.columns).difference(exog)]
        test_exog = test_series[exog]
    else:
        test_series = test_series[test_series.columns[0]]
        test_exog = None

    forecast, conf = model.predict(len(test_series), exogenous=test_exog, return_conf_int=True, alpha=params['alpha'])
    accuracy = forecast_accuracy(forecast, np.array(test_series), err_metric=err_metric)

    return accuracy, forecast


def train_model(train, params, exog=None):

    if exog:
        train_exog = train[set(train.columns).difference(exog)]
        train_series = train[set(train.columns).difference(exog)]

    else:
        train_series = train[train.columns[0]]
        train_exog = None

    model = pm.auto_arima(train_series, exogenous=train_exog, trace=True,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True, **params)

    print(model.summary())

    return model


def train_test(train, test, params, model_output_file,  exog=None,  err_metric='all'):

    if exog:
        train_exog = train[exog]
        test_exog = test[exog]

        train_series = train[set(train.columns).difference(exog)]
        test_series = test[set(train.columns).difference(exog)]

    else:

        train_series = train[train.columns[0]]
        test_series = test[test.columns[0]]

        train_exog = None
        test_exog = None

    model = pm.auto_arima(train_series, exogenous=train_exog, trace=True,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True, **params)

    print(model.summary())
    print("Dumped the model at:", model_output_file)
    dump_model(model, model_output_file)

    forecast, conf = model.predict(len(test), exogenous=test_exog, return_conf_int=True, alpha=params['alpha'])
    accuracy = forecast_accuracy(forecast, np.array(test_series), err_metric=err_metric)

    return accuracy, forecast


def get_params(parsed_args):

    params = {'start_p': parsed_args.start_p,
              'start_q': parsed_args.start_q,
              'max_p': parsed_args.max_p,
              'max_d': parsed_args.max_d,
              'max_q': parsed_args.max_q,
              'd': parsed_args.d,
              'm': parsed_args.m,
              'D': parsed_args.D,
              'start_P': parsed_args.start_P,
              'start_Q': parsed_args.start_Q,
              'max_P': parsed_args.max_P,
              'max_Q': parsed_args.max_Q,
              'max_D': parsed_args.max_D,
              'seasonal': parsed_args.seasonal,
              'seasonal_test': parsed_args.seasonal_test,
              'stationary': parsed_args.stationary,
              'information_criterion': parsed_args.information_criterion,
              'alpha': parsed_args.alpha,
              'test': parsed_args.test,
              'trend': parsed_args.trend,
              'method': parsed_args.method,
              'transparams': parsed_args.transparams,
              'solver': parsed_args.solver,
              'maxiter': parsed_args.maxiter,
              'disp': parsed_args.disp,
              'out_of_sample_size': parsed_args.out_of_sample_size,
              }
    return params


if __name__ == "__main__":

    main()
