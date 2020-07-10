

from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dateparser import parse as parse_date
from functools import partial
import pandas as pd
import numpy as np
import argparse


def main():

    parser = argparse.ArgumentParser()

    # Method Argument
    parser.add_argument("--verbose", action='store_true', help="verbose : bool, optional, default False", default=False)

    parser.add_argument("--run_test", action='store_true', default=False, help="Runs test")
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
                        help="Time series data file (csv only) path ",
                        action='store')

    parser.add_argument("--train_series",
                        help="training time series data file (csv only) path ",
                        action='store')
    parser.add_argument("--test_series",
                        help="testing time series data file (csv only) path ",
                        action='store')
    parser.add_argument("--n_predict_periods",
                        help="number of future time periods for which prediction should be made",
                        type=int,
                        default=1,
                        action='store')

    parser.add_argument("-S", "--train_test_split",
                        help="Fraction of data to be used for training the model "
                             "required if train test data is not pre split",
                        type=float,
                        default=0.8,
                        action='store')
    parser.add_argument("--pre_split",
                        help="Boolean Indicator variable, which tells if the train test data is splitted beforehand",
                        default=False,
                        action='store_true')

    parser.add_argument("--date_index",
                        help="Column name containing dates, which will be used as the data index",
                        type=str,
                        # required=True,
                        action='store')
    parser.add_argument("--date_format",
                        help="Give the date format in data",
                        type=str,
                        default="%Y-%m-%d",
                        action='store')
    parser.add_argument("--endog",
                        help="Column name containing endogenous variable",
                        type=str,
                        nargs='+',
                        # required=True,
                        action='store')

    parser.add_argument("--exog",
                        help="Optional, Column name containing exogenous variable",
                        type=str,
                        nargs='*',
                        required=False,
                        action='store')

    # Model Parameter Arguments
    parser.add_argument("--error_metric",
                        help="Error metric to be used. Available options are "
                             "'mse'(mean squared error), 'mae' (mean absolute error) and 'rmse' ",
                        default='mse',
                        action='store')

    parser.add_argument("-p", help="AR parameters",
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument("-d",
                        help="Difference parameters",
                        type=int,
                        default=0,
                        action='store')
    parser.add_argument("-q", help="MA parameters",
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument("--freq",
                        help="str, optional"
                             "The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',"
                             "'M', 'A', or 'Q'. This is optional if dates are given.",
                        action='store')

    # Model fitting parameters
    parser.add_argument("--transparams",
                        help = "bool, optional "
                               "Whehter or not to transform the parameters to ensure stationarity. "
                               "If False, no checking for stationarity or invertibility is done.",
                        default=False,
                        action='store_true')
    parser.add_argument("--method",
                        help="method : str {'css-mle','mle','css'}"
                             "This is the loglikelihood to maximize.  If 'css-mle'', the "
                             "conditional sum of squares likelihood is maximized and its values "
                             "are used as starting values for the computation of the exact "
                             "likelihood via the Kalman filter.  If 'mle', the exact likelihood "
                             "is maximized via the Kalman Filter.  If 'css' the conditional sum "
                             "of squares likelihood is maximized.  All three methods use "
                             "`start_params` as starting parameters.",
                        type=str,
                        default='css-mle',
                        action='store')

    parser.add_argument("--trend",
                        help="str {'c','nc'} "
                             "Whether to include a constant or not.  'c' includes constant, "
                             "'nc' no constant.",
                        type=str,
                        default='c',
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
                        help="int, optional "
                             "The maximum number of function evaluations. Default is 500.",
                        type=int,
                        default=500,
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
                        default=5,
                        action='store')
    parser.add_argument("--start_ar_lags",
                        help="int, optional "
                             "Parameter for fitting start_params. When fitting start_params, "
                             "residuals are obtained from an AR fit, then an ARMA(p,q) model is "
                             "fit via OLS using these residuals. If start_ar_lags is None, fit "
                             "an AR process according to best BIC. If start_ar_lags is not None, "
                             "fits an AR process with a lag length equal to start_ar_lags.",
                        default=None,
                        action='store')


    parsed_args = parser.parse_args()

    # Checking for necessary input arguments in each case

    if parsed_args.run_test:

        parsed_args.p = 5
        parsed_args.d = 1
        parsed_args.q = 0

        parsed_args.endog = ['Sales']
        parsed_args.date_index = ['Month']

        series = load_data('./data/ts_data.csv', date_column='Month')
        print("Data Loaded with {} rows and {} columns".format(series.shape[0], series.shape[1]))
        assert series.shape[0] == 36, "Test failed, row counts didn't match"
        assert series.shape[1] == 1, "Test failed, column counts didn't match"

        train, test = train_test_split(series, split_perc=0.66)
        print("Data split into train ({} rows) and test ({} rows)".format(train.shape[0], test.shape[0]))
        assert train.shape[0] == 23, "Test failed, train rows counts didn't match"
        assert test.shape[0] == 13, "Test failed, test rows counts didn't match"

        model_params, fit_params = get_params(parsed_args)
        print("Model Parameters : \n", model_params)
        print("Model Fit Parameters : \n", fit_params)

        test_score, predictions = train_test_model(train, test, model_params, fit_params, './models/arima_test.pkl', err_metric=parsed_args.error_metric)

        print("Model Trained")
        print("Model Tested over test data, Test score ('mse') is :", test_score)
        assert test_score == 6438.015940500031, "Test Failed"
        print("Test Successful")

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

        model_params, fit_params = get_params(parsed_args)

        test_score, predictions = train_test_model(train_series, test_series, model_params,
                                                   fit_params, parsed_args.model_output_path,
                                                   err_metric=parsed_args.error_metric)

        print("The model is dumped at following location:", parsed_args.model_output_path)
        print("Test Score ({}): ".format(parsed_args.error_metric), test_score)

    elif parsed_args.predict:
        assert parsed_args.input_model, "Please provide an Input Model Path"
        assert parsed_args.n_predict_periods, "Please provide number of periods for prediction"
        assert parsed_args.result_path, "Please provide output result file path"

        model = load_model(parsed_args.input_model)
        predicted = model.forecast(steps=parsed_args.n_predict_periods)
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


def train_test_model(train_series, test_series, model_params, fit_params, model_output_path, err_metric='mse'):
    p,d,q = model_params['p'], model_params['d'], model_params['q']

    train_endo_factors = np.array(train_series[model_params['endog']])
    test_endo_factors = np.array(test_series[model_params['endog']])

    history_endo = [x for x in train_endo_factors]
    predictions = list()

    if model_params['exog']:
        train_exog_factors = np.array(train_series[model_params['exog']])
        test_exog_factors = np.array(test_series[model_params['exog']])


        history_exog = [x for x in train_exog_factors]
        model = ARIMA(history_endo, order=(p, d, q), exog=history_exog, freq=model_params['freq'])
        model_fit = model.fit(**fit_params)
        model_fit.save(model_output_path)


        for t in range(len(test_endo_factors)):
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs_endo = test_endo_factors[t]
            obs_exog = test_exog_factors[t]

            history_endo.append(obs_endo)
            history_exog.append(obs_exog)

            model = ARIMA(history_endo, order=(p, d, q), exog=history_exog, freq=model_params['freq'])
            model_fit = model.fit(**fit_params)
            print('predicted=%f, expected=%f' % (yhat, obs_endo))

    else:
        model = ARIMA(history_endo, order=(p, d, q), freq=model_params['freq'])
        model_fit = model.fit(**fit_params)
        model_fit.save(model_output_path)

        for t in range(len(test_endo_factors)):

            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_endo_factors[t]
            history_endo.append(obs)

            model = ARIMA(history_endo, order=(p, d, q), freq=model_params['freq'])
            model_fit = model.fit(**fit_params)

            print('predicted=%f, expected=%f' % (yhat, obs))

    error = calculate_error(test_endo_factors, predictions, error_metric=err_metric)
    return error, predictions






# def train_model(train_series, model_params, fit_params, model_output_path):
#     p, d, q = model_params['p'], model_params['d'], model_params['q']
#
#     train_endo_factors = np.array(train_series[model_params['endog']])
#     history_endo = [x for x in train_endo_factors]
#
#     if model_params['exog']:
#         train_exog_factors = np.array(train_series[model_params['exog']])
#
#         history_exog = [x for x in train_exog_factors]
#         model = ARIMA(history_endo, order=(p, d, q), exog=history_exog, freq=model_params['freq'])
#         model_fit = model.fit(**fit_params)
#         model_fit.save(model_output_path)
#
#     else:
#         model = ARIMA(history_endo, order=(p, d, q), freq=model_params['freq'])
#         model_fit = model.fit(**fit_params)
#         model_fit.save(model_output_path)
#
#     return model_fit


def calculate_error(observed, predicted, error_metric='mse'):
    if error_metric == 'mse':
        err = mean_squared_error(observed, predicted)
    elif error_metric == 'mae':
        err = mean_absolute_error(observed, predicted)
    elif error_metric == 'rmse':
        err = np.sqrt(mean_squared_error(observed, predicted))
    return err


def load_model(model_path):
    return ARIMAResults.load(model_path)


def load_data(data_path, date_column, date_format):
    parse_dt = partial(parse_date, date_formats=date_format)
    return pd.read_csv(data_path, date_parser=parse_dt, index_col=date_column, parse_dates=date_column)


def get_params(parsed_args):

    model_params = {
            'p': parsed_args.p,
            'd': parsed_args.d,
            'q': parsed_args.q,
            'exog': parsed_args.exog,
            'endog': parsed_args.endog,
            'date_column': parsed_args.date_index,
            'freq' : parsed_args.freq
            }

    fit_params = {
        'trend': parsed_args.trend,
        'method': parsed_args.method,
        'transparams': parsed_args.transparams,
        'solver' : parsed_args.solver,
        'maxiter': parsed_args.maxiter,
        'tol': parsed_args.tol,
        'disp': parsed_args.disp,
        'start_ar_lags': parsed_args.start_ar_lags
    }
    return model_params, fit_params


if __name__ == "__main__":
    main()
