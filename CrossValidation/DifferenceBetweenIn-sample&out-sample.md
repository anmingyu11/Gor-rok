By the "sample" it is meant the data sample that you are using to fit the model.

First - you have a sample
Second - you fit a model on the sample
Third - you can use the model for forecasting

If you are forecasting for an observation that was part of the data sample - it is in-sample forecast.

If you are forecasting for an observation that was not part of the data sample - it is out-of-sample forecast.

So the question you have to ask yourself is: **Was the particular observation used for the model fitting or not ? If it was used for the model fitting, then the forecast of the observation is in-sample. Otherwise it is out-of-sample.**

> if you use data 1990-2013 to fit the model and then you forecast for 2011-2013, it's in-sample forecast. but if you only use 1990-2010 for fitting the model and then you forecast 2011-2013, then its out-of-sample forecast.