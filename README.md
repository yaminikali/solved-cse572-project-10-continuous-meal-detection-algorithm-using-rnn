Download Link: https://assignmentchef.com/product/solved-cse572-project-10-continuous-meal-detection-algorithm-using-rnn
<br>
Continuous Glucose Monitoring Prediction Method

<em>Abstract</em>—We constructed and analyzed algorithms that developed a continuous meal detection method. We were given glucose monitor and bolus data for an individual over the course of six months in five minute intervals. First, we parsed the CGM data into two hours intervals and synchronized the CGM data with meal ground truth. Next, we developed two algorithms: 1) an auto regression based model and 2) a recurrent neural network based model. We then applied our models to a new patient. Lastly, we performed an execution time analysis comparison of the methods.

<ol>

 <li>INTRODUCTION</li>

</ol>

The advances in machine learning and data modeling have opened up incredible opportunities for new applications. Continuous glucose monitoring is a system that automatically track blood glucose levels throughout the day. The CGM works through a small sensor underneath the skin. The sensor sends the data to an external device where data can be retrieved and stored. Whenever you eat glucose levels spike, then settle back down in the coming minutes. Our goal was to come up with an efficient algorithm that detects when a meal is being eaten. This information is very important and has a variety of applications, including diabetes management among other things.

<ol>

 <li>AUTHOR KEYWORDS</li>

</ol>

CGM, RNN, Auto Regression, Bolus

<ul>

 <li>PROJECT SETUP</li>

</ul>

The first part of the project was setting up the data. We were provided with six months worth of CGM sensor outputs data for a given person. In a separate file, we were given Bolus data with meal ground truth. If the Bolus value was high then the person was eating a meal at that time. If the value was low, they were not eating a meal at that time. Before we could train our models, we needed to parse the CGM data and synchronize the meal ground truth with the CGM data. I parsed the CGM data inside python which I will discuss in the next section. To synchronize the meal ground with the CGM data I simply joined the two columns together based off of the time column. Since the two time columns were the same for both variables, this automatically synchronized meal ground truth with the CGM data.

Next we decided on the programming language to write our models in. We decided to user Python since Python supports a variety of powerful machine learning libraries such as Tensor Flow for RNN and a library for SARIMA, used in the auto regression model. In addition, our group was very comfortable coding in Python, making the language easy to work with.

<ol>

 <li>IMPLEMENTATION</li>

 <li><em> RNN based meal detection</em></li>

</ol>

<ul>

 <li><em>CGM Data Parsing and Data Preperation: </em>The first step was to import the Bolus and CGM data into a Python script. The data had been synchronized in Excel, and the Bolus data had been categorized as 0 (not a meal) or .99 (a meal) in Excel as well. Next I got rid of any null values by deleting any rows with null values. This did not decrease the dataset by a considerable amount, otherwise I would have interpolated the data. Next I created a series to supervised function that parsed the CGM data into two hour increments. This function helped my RNN model learn off of two hour intervals. Next I prepared by data to be fed into the RNN model. I feature scaled the data so it was normalized, split the data into x and y variables, and split it into a training and test set. I found that an 85/15 train-test split gave the most accurate results. I then reshaped the data into 3 dimensions: samples, timestamps, and features. This is the input shape for an RNN algorithm. The next step was the RNN algorithm development.</li>

 <li><em>Algorithm Instantiation: </em>The algorithm was developed using the Keras library inside of Tensorflow. I used the Sequential model inside Keras since we had one input (CGM sensor data) and one output (meal ground truth).</li>

 <li><em>Algorithm Development: </em>I decided to use a Long Short Term Memory (LSTM), a type of Recurrent Neural Network (RNN), since LSTM’s are powerful when classifying time series data [1]. It also handles lag between important events, such as having null values in a dataset, with ease. I added a dropout layer with a value of .4 and two dense layers. The first dense layer had a ”tanh” activation function while the second dense layer had a ”sigmoid” activation function. I trained many models an found this setup gave the most accurate results in terms of training and testing accuracy. The loss function used was ”categorial crossentropy” since we are classifying the data as a meal or not.</li>

 <li><em>Algorithm Implementation: </em>I fit the model on the training data and validated it with the test data. To reiterate, the data was split into training and testing data previously in the program. The first 85 percent of rows was considered training data, and the last 15 percent as test data. I found fitting the model on 3 epochs with a batch size of 50 gave the most accurate results.</li>

 <li><em>Training and Testing accuracy: </em>I evaluated the training and testing accuracy using the evaluate function of the keras models library. The function found the RNN model has a training accuracy of 95.4 percent and a testing accuracy of 95.9 percent.</li>

 <li><em>Applying RNN Algorithm to New Patient: </em>To apply the RNN Algorithm to a new patient, I first reshaped the data. The training data remained the same, but the test data was the new data given for the new patient. This training and new test data was fit to the model, again using 3 epochs and a batch size of 50. Using the evaluate method in the models library, the training accuracy was 95.9 percent and the testing accuracy was 96.7 percent.</li>

 <li><em>Execution Time Analysis of RNN Algorithm: </em>I used the ”timeit” method in Python to retrieve an execution time for the algorithm. I made a function that simply ran the algorithm once, ensuring timeit didn’t time any other line of code that could take a considerable amount of time to run. Timeit found the RNN Algorithm model took, on average, 5.67 seconds to run. This is a long time and could be do to the complicated LSTM RNN used. In addition, the dataset was not small. Lastly, the batch size and epoch variables could have contributed to the time as well. Adjusting these numbers could lower the time it takes for this model to run.</li>

</ul>

<ol>

 <li><em> Auto regression based modeling of CGM</em></li>

</ol>

<ul>

 <li><em>SARIMA Algorithm: </em>The Seasonal Auto-Regressive Integrated Moving Average (SARIMA) model is the same model as ARIMA but it also depends on the seasonality of the data, which is defined with the addition of P, D, and Q for the seasonal portion of the time series and the <em>m </em>is the number of observations per period of time. So the model is as follows:</li>

</ul>

(1)

non-seasonal        seasonal

For the development of the SARIMA model, the statsmodels python package was used as well as Jupyter Notebook, Pandas, and NumPy. Statsmodels[2] provides a helpful setup instructions and examples using SARIMAX model, which was taken as the baseline for the code implementation provided.

<ul>

 <li><em>Algorithm Instantiation: </em>This section will describe the setup that was used to apply the SARIMAX model to our CGM data. The analysis done from this point on follows the template provided by statsmodel[2] and Kaggle[3], and was used purely as a reference tool to guide through the analysis and model application. First, to get a good grasp on how the data correlates plotting it is beneficial, and we end up with the following:</li>

</ul>

Fig. 1. CGM and Bolus data plot.

The data is erratic and is clearly not stationary, as the variance and mean are not constant through time, which is an indication of <em>heteroscedasticity</em>. As the usual plot did not provide any data, we can plot Partial Auto-Correlation Function (PACF, Figure 2) and Complete Auto-Correlation Function (ACF, Figure 3) to check there is any information to be obtained.

Fig. 2. Partial Auto-Correlation Function (PACF) plot.

Fig. 3. Complete Auto-Correlation Function (ACF) plot.

Again, similarly to the Figure 1, but apart from the data that was already obtained before nothing stands out. Assuming that the time series is non-stationary we can apply the logarithmic difference in order to make it stationary, we obtain the Figure

4.

Fig. 4. Logarithmic difference of CGM data plot.

The application of the logarithmic difference has proved to be correct and the plot (see Figure 4) is stationary, but the is still the indication of seasonality of the data. Thus, if we apply the periodic nature of the data to the plot we will obtain Figure 5.

Fig. 5. Logarithmic difference of CGM data plot with periodic difference.

To test that the data is stationary we apply the DickeyFuller test, and see that the p-value is not significant enough to accept the null hypothesis, therefore we can confirm that the data is indeed stationary. Now, applying PACF (see Figure 2) and ACF (see Figure 7) on the adjusted data, from Figure 6 we can see that the a spike at <em>1</em>, thus indicating the seasonal auto-regressive process of order <em>AR(1) </em>and <em>P = 1</em>. Looking at the ACF plot (see Figure 7), we can see a spike at <em>1</em>, thus indicating the non-seasonal <em>MA(1)</em>. This provides us with the rough idea of the processes that correlet to our SARIMA model, but applying the optimize SARIMA function results in the <em>SARIMA(0,1,2)(0,1,2,4) </em>best fit scenario.

<ul>

 <li><em>Algorithm Implementation: </em>Now, after obtaining the best fit processes for the SARIMA model, we can apply the algorithm using the statsmodel’s built in SARIMAX function. Which provides us with the results shown in Figure 8, which provides us with a model that contains both seasonal and nonseasonal processes.</li>

 <li><em>Training and Testing accuracy: </em>After obtaining the model, we can plot the residuals (see Figure 9). Inspecting the QQ plot, we can see observe a nearly straight line, suggesting the absence of systematic departure from normality. Taking a closer look at correlation statistics (Correlogram) implies that there is no auto-correlation between the residuals.</li>

</ul>

Thus, allowing us to plot the model along with the forecast (see Figure 10). As the plot shows a tight prediction and correlation between the model and the actual data, thus implying a positive forecast result. Evaluating the results, the model

Fig. 6. Partial Auto-Correlation Function (PACF) plot.

Fig. 7. Complete Auto-Correlation Function (ACF) plot.

Fig. 8. Statespace Model Results.

Fig. 9. Statespace Model Results.

accuracy results in 91.5 percent and the execution time being

1.35 seconds.

Fig. 10. Model forecast.

<ol>

 <li>COMPLETION OF TASKS</li>

</ol>

<table width="334">

 <tbody>

  <tr>

   <td width="36">Task</td>

   <td width="205">Description</td>

   <td width="93">Assignee</td>

  </tr>

  <tr>

   <td width="36">1</td>

   <td width="205">Parse CGM Data</td>

   <td width="93">Josh O’Callaghan</td>

  </tr>

  <tr>

   <td width="36">2</td>

   <td width="205">Synchronize meal ground truth with CGM Data</td>

   <td width="93">Alex Pappas</td>

  </tr>

  <tr>

   <td width="36">3</td>

   <td width="205">Develop Auto Regression Model</td>

   <td width="93">Daniar Tabys</td>

  </tr>

  <tr>

   <td width="36">4</td>

   <td width="205">Develop RNN Model</td>

   <td width="93">Sam Steinberg</td>

  </tr>

  <tr>

   <td width="36">5</td>

   <td width="205">Apply algorithms to new patient</td>

   <td width="93">Alex Pappas</td>

  </tr>

  <tr>

   <td width="36">6</td>

   <td width="205">Execution time analysis</td>

   <td width="93">Josh O’Callaghan</td>

  </tr>

 </tbody>

</table>

<ol>

 <li>LIMITATIONS</li>

</ol>

While the project was a success, there were some limitations. First off, some of the data contained null values. This required interpolation, meaning the the average of the values immediately before and after the null value(s) is taken. This new value fills in the null value(s). Our models may have been more accurate provided we had data with fewer null values.

The meal ground truth in the training/test set was another limitation. There were no specific guidelines on which bolus data was a meal and what wasn’t, other than high numbers are meals and low numbers are not. Usually the values were easy to classify, since most of the numbers were extremely low (¡0.25), while others were quite high (¡1). Unfortunately there were some values between 0.25 and 1, making it difficult to classify them. Thankfully, these were the only limitations in the project.

<ul>

 <li>CONCLUSION</li>

</ul>

In this project, we made two algorithms to implement a continuous glucose monitoring method. We were given CGM sensor and meal ground truth data from an individual over the course of six months. The two algorithms were an Auto Regression based model and a Recurrent Neural Network based model. The Auto Regression was made using SARIMA and the RNN by libraries in Tensor Flow, both coded in Python. The training CGM data was parsed so the models learned in two hours samples of the data. The dataset was then synchronized with the meal ground truth data. Training and testing accuracy was gathered and compared with both models. The RNN was slightly more accurate than the auto regression model. Next, our algorithms were applied to a new patient to verify the models were not over fitting the training data. Lastly, execution times were taken from both algorithms and compared. The auto-regression was more efficient with an average run time of only 1.35 seconds compared to the 5.67 seconds of the RNN. Since the auto regression algorithm takes considerably less time to run and still has a very high accuracy (training and test both above 90 percent), I would choose the auto-regression model. Even so, both of these algorithms can be used to detect when a person is eating a meal which spikes glucose levels. This information is very useful as it is used in a variety of applications including diabetes management.

<ul>

 <li>ACKNOWLEDGEMENTS</li>

</ul>

We would like to thank Professor Banerjee for giving us the opportunity to work on this project. We are all very interested in machine learning and classification, so the project aligned well with our interests. In addition, glucose monitoring impacts a large amount of people. Being able to use our passions of machine learning to have an impact on a real world problem was fulfilling.

REFERENCES

<ul>

 <li>Mittal, Aditi. <em>Understanding RNN and LSTM</em>. https:// www.towardsdatascience.com/understanding-rnn-andlstm-f7cdf6dfc14e, Last accessed on 2020-08-13. 2019.</li>

 <li>Perktold, Seabold, Taylor, J. P. S. S. J. T. (n.d.) <em>SARIMAX: Introduction</em>. https://www.statsmodels.org/dev/ examples/notebooks/generated/statespace sarimax stata. html, Last accessed on 2020-08-13.</li>

 <li><em>How to use SARIMAX</em>. https://www.kaggle.com/ poiupoiu/how-to-use-sarimax, Last accessed on 202008-13. 2018.</li>

</ul>