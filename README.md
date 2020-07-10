# ts_cl_wrapper


### Auto ARIMA -- Command Line Usage 

	Argument	Description
	 -h, --help            	show this help message and exit
	  --verbose            	 verbose : bool, optional, default False
	  --run_test            	Runs test
	  --train_test         	 Boolean argument for running training and testing simultaneously
	  --training            	Boolean argument for running training
	  --testing             	Boolean argument for running testing
	  --predict            	 Boolean argument for running the predictions on a pretrained model. Model input file is neccessary, Returns predicted values. and stores them in result file
	  --result_path RESULT_PATH	                        Result output file path.
	  --input_model INPUT_MODEL	                        Input Model Address
	  --model_output_path MODEL_OUTPUT_PATH	                        Output Model File Path
	  --series SERIES       	Time series data file (csv only) path
	  --train_series TRAIN_SERIES	                        training time series data file (csv only) path
	  --test_series TEST_SERIES	                        testing time series data file (csv only) path
	  --n_predict_periods N_PREDICT_PERIODS	number of future time periods for which prediction should be made
	  -S TRAIN_TEST_SPLIT, --train_test_split TRAIN_TEST_SPLIT	                        Fraction of data to be used for training the model required if train test data is not pre split
	  --pre_split           	Boolean Indicator variable, which tells if the train test data is splitted beforehand
	  --date_index DATE_INDEX	Column name containing dates, which will be used as the data index
	  --date_format DATE_FORMAT	Give the date format in data
	  --endog ENDOG [ENDOG ...]	 Column name containing endogenous variable
	  --exog [EXOG [EXOG ...]]	Optional Column name containing exogenous variable
	  --error_metric ERROR_METRIC	 Error metric to be used. Available options are 'mse'(mean squared error), 'mae' (mean absolute error) and 'rmse'
	  --verbose             verbose : bool, optional, default False	
	  --error_metric ERROR_METRIC	String, Default 'all', Error metric to be used. Available options are 'mse'(mean squared error), 'mae' (mean absolute error), 'rmse', (root mean square error) 'me' (mean error), 'mpe' (mean  'maxs', above)'minmax', 'acf1', 'all' (Calculates all of thepercentageerror), 'corr' (error correlation coeficient),'mins', 
	  --freq FREQ          	 str, optionalThe frequency of the time-series. A Pandas offset or 'B', 'D', 'W','M', 'A', or 'Q'. This is optional if dates are given.
	  --no_transparams         	bool flag stores False, optional (default True) Whehter or not to transform the parameters to ensure stationarity. If False, no checking for stationarity or invertibility is done.
	  --seasonal            bool, optional, (default False)	
	  --m M                 int, optional (default=1) The period for seasonal	
	                        differencing, 'm' refers to the number of periods in	
	                        each season. For example, 'm' is 4 for quarterly data,	
	                        12 for monthly data, or 1 for annual (non-seasonal)	
	                        data. Default is 1. Note that if 'm' == 1 (i.e., is	
	                        non-seasonal), ''seasonal'' will be set to False.	
	  --stationary          bool, optional, (default False)	
	  --trend TREND         str {'c','nc'}, (default None) Whether to include a	
	                        constant or not. 'c' includes constant, 'nc' no	
	                        constant.	
	  --d D                 int, optional, (default None) The order of first-	
	                        differencing. If None (by default), the value will	
	                        automatically be selected based on the Augmented	
	                        Dickey-Fuller or the Phillips Perron test will be	
	                        conducted to find the most probable value). Must be a	
	                        positive integer or None.	
	  --start_p START_P     int, optional, (default 1) The starting value of 'p',	
	                        the order (or number of time lags) of the auto-	
	                        regressive ('AR') model. Must be a positive integer.	
	  --start_q START_Q     int, optional, (default 1) The starting value of 'q',	
	                        the order of the moving-average ('MA') model. Must be	
	                        a positive integer.	
	  --max_p MAX_P         int, optional, (default 5) The maximum value of 'p',	
	                        inclusive. Must be a positive integer greater than or	
	                        equal to 'start_p'.	
	  --max_q MAX_Q         int, optional, (default 5) The maximum value of 'q',	
	                        inclusive. Must be a positive integer greater than	
	                        'start_q'.	
	  --max_d MAX_D         int, optional, (default 2) The maximum value of 'd',	
	                        inclusive. Must be a positive integer greater than	
	                        'd'.	
	  --D D                 int, optional (default=None)The order of the seasonal	
	                        differencing. If None (by default, the valuewill	
	                        automatically be selected based on the results of	
	                        the'seasonal_test'. Must be a positive integer or	
	                        None.	
	  --start_P START_P     int, optional (default=1) The starting value of 'P',	
	                        the order of the auto-regressive portion of the	
	                        seasonal model.	
	  --start_Q START_Q     int, optional (default=1) The starting value of 'Q',	
	                        the order of the moving-average portion of the	
	                        seasonal model.	
	  --max_P MAX_P         int, optional (default=2) The maximum value of 'P',	
	                        inclusive. Must be a positive integer greater than	
	                        'start_P'.	
	  --max_Q MAX_Q         int, optional (default=2) The maximum value of 'Q',	
	                        inclusive. Must be a positive integer greater than	
	                        'start_Q'.	
	  --max_D MAX_D         int, optional (default=1) The maximum value of ''D''.	
	                        Must be a positive integer greater than 'D'.	
	  --information_criterion INFORMATION_CRITERION	
	                        str, optional (default='aic') The information	
	                        criterion used to select the best ARIMA model. Options	
	                        are ('aic', 'bic', 'hqic', 'oob').	
	  --alpha ALPHA         float, optional (default=0.05) Level of the test for	
	                        testing significance.	
	  --stationary_test TEST	
	                        str, optional (default='kpss') Type of unit root test	
	                        to use in order to detect stationarity if	
	                        ''stationary'' is False and ''d'' is None. Default is	
	                        'kpss' (Kwiatkowski Phillips Schmidt Shin). Other	
	                        option is 'adf' (Augmented Dickey-Fuller)	
	  --seasonal_test SEASONAL_TEST	
	                        str, optional (default='ocsb') This determines which	
	                        seasonal unit root test is used if ''seasonal'' is	
	                        True and ''D'' is None. Default is 'OCSB'.	
	  --method METHOD       method : str {'css-mle','mle','css'}, (default	
	                        None)This is the loglikelihood to maximize. If 'css-	
	                        mle'', the conditional sum of squares likelihood is	
	                        maximized and its values are used as starting values	
	                        for the computation of the exact likelihood via the	
	                        Kalman filter. If 'mle', the exact likelihood is	
	                        maximized via the Kalman Filter. If 'css' the	
	                        conditional sum of squares likelihood is maximized.	
	                        All three methods use 'start_params' as starting	
	                        parameters.	
	  --solver SOLVER       str or None, optional Solver to be used. The default	
	                        is 'lbfgs' (limited memory Broyden-Fletcher-Goldfarb-	
	                        Shanno). Other choices are 'bfgs', 'newton' (Newton-	
	                        Raphson), 'nm' (Nelder-Mead), 'cg' - (conjugate	
	                        gradient), 'ncg' (non-conjugate gradient), and	
	                        'powell'. By default, the limited memory BFGS uses	
	                        m=12 to approximate the Hessian, projected gradient	
	                        tolerance of 1e-8 and factr = 1e2. You can change	
	                        these by using kwargs.	
	  --maxiter MAXITER     int, optional (default=None) The maximum number of	
	                        function evaluations. Statsmodels defaults this value	
	                        to 50 for SARIMAX models and 500 for ARIMA and ARMA	
	                        models. If passed as None, will use the seasonal order	
	                        to determine which to use (50 for seasonal, 500	
	                        otherwise).	
	  --out_of_sample_size OUT_OF_SAMPLE_SIZE	
	                        int, optional (default=0) The ''ARIMA'' class can fit	
	                        only a portion of the data if specified, in order to	
	                        retain an 'out of bag' sample score. This is the	
	                        number of examples from the tail of the time series to	
	                        hold out and use as validation examples. The model	
	                        will not be fit on these samples, but the observations	
	                        will be added into the model's 'endog' and 'exog'	
	                        arrays so that future forecast values originate from	
	                        the end of the endogenous vector.	
	  --scoring SCORING     str, optional (default='mse') If performing validation	
	                        (i.e., if ''out_of_sample_size'' > 0), the metric to	
	                        use for scoring the out-of-sample data. One of ('mse',	
	                        'mae')	
	  --tol TOL             tol : float The convergence tolerance. Default is	
	                        1e-08.	
	  --disp DISP           int, optional If True, convergence information is	
	                        printed. For the default l_bfgs_b solver, disp	
	                        controls the frequency of the output during the	
	                        iterations. disp < 0 means no output in this case.	


### ARIMA -- Command Line Usage 

	Argument	Description
	 -h, --help            	show this help message and exit
	  --verbose            	 verbose : bool, optional, default False
	  --run_test            	Runs test
	  --train_test         	 Boolean argument for running training and testing simultaneously
	  --predict            	 Boolean argument for running the predictions on a pretrained model. Model input file is neccessary, Returns predicted values. and stores them in result file
	  --result_path RESULT_PATH	                        Result output file path.
	  --input_model INPUT_MODEL	                        Input Model Address
	  --model_output_path MODEL_OUTPUT_PATH	                        Output Model File Path
	  --series SERIES       	Time series data file (csv only) path
	  --train_series TRAIN_SERIES	                        training time series data file (csv only) path
	  --test_series TEST_SERIES	                        testing time series data file (csv only) path
	  --n_predict_periods N_PREDICT_PERIODS	number of future time periods for which prediction should be made
	  -S TRAIN_TEST_SPLIT, --train_test_split TRAIN_TEST_SPLIT	                        Fraction of data to be used for training the model required if train test data is not pre split
	  --pre_split           	Boolean Indicator variable, which tells if the train test data is splitted beforehand
	  --date_index DATE_INDEX	Column name containing dates, which will be used as the data index
	  --date_format DATE_FORMAT	Give the date format in data
	  --endog ENDOG [ENDOG ...]	 Column name containing endogenous variable
	  --exog [EXOG [EXOG ...]]	Optional Column name containing exogenous variable
	  --error_metric ERROR_METRIC	 Error metric to be used. Available options are 'mse'(mean squared error), 'mae' (mean absolute error) and 'rmse'
	  -p P                  	AR parameters
	  -d D                  	Difference parameters
	  -q Q                  	MA parameters
	  --freq FREQ          	 str, optionalThe frequency of the time-series. A Pandas offset or 'B', 'D', 'W','M', 'A', or 'Q'. This is optional if dates are given.
	  --transparams         	bool, optional Whehter or not to transform the parameters to ensure stationarity. If False, no checking for stationarity or invertibility is done.
	  --method METHOD       	method : str {'css-mle','mle','css'} This is the loglikelihood to maximize. If 'css-mle'', the conditional sum of squares likelihood is maximized and its values are used as starting values for the computation of the exact likelihood via the Kalman filter. If 'mle', the exact likelihood is maximized via the Kalman Filter. If 'css' the conditional sum of squares likelihood is maximized. All three methods use `start_params` as starting parameters.
	  --trend TREND         	str {'c','nc'} Whether to include a constant or not.  'c' includes constant, 'nc' no constant.
	  --solver SOLVER       	str or None, optional Solver to be used. The default is 'lbfgs' (limited memory Broyden-Fletcher-Goldfarb-Shanno). Other choices are 'bfgs', 'newton' (Newton-Raphson), 'nm' (Nelder-Mead), 'cg' - (conjugate gradient), 'ncg' (non-conjugate gradient), and 'powell'. By default, the limited memory BFGS uses m=12 to approximate the Hessian, projected gradient tolerance of 1e-8 and factr = 1e2. You can change these by using kwargs.
	  --maxiter MAXITER     	int, optional The maximum number of function evaluations. Default is 500.
	  --tol TOL            	 tol : float The convergence tolerance. Default is 1e-08.
	  --disp DISP          	 int, optional If True, convergence information is printed. For the default l_bfgs_b solver, disp controls the frequency of the output during the iterations. disp < 0 means no output in this case.
	  --start_ar_lags START_AR_LAGS	int, optional Parameter for fitting start_params. When fitting start_params, residuals are obtained from an AR fit, then an ARMA(p,q) model is fit via OLS using these residuals. If start_ar_lags is None, fit an AR process according to best BIC. If start_ar_lags is not  None, fits an AR process with a lag length equal to start_ar_lags.