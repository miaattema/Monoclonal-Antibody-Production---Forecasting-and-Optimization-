
## Monoclonal Antibody Production – Optimization through Time Series Forecasting 
This project allows for the evaluation of linear and non-linear vector autoregression models, using a variable number of lagged values to predict current values by time series forecasting. The final model is chosen based on the lowest error, simplicity, and white noise criterion; it is optimized to find the best initial conditions such that space time profit and space time yield are highest. 
## Highlights
	- Developed linear and nonlinear vector autoregression models to forecast time-dependent bioreactor conditions.
	- Evaluated regression processes and model hyperparameters using cross-validation and RMSE.
	- Optimized initial culture conditions to maximize space-time profit and mAb space-time yield.
	- Analyzed the influence of glucose, glutamine, and viable cell density on bioprocess performance.
## Environment
	-Python (version 3.14.7) 
    -matplotlib (version 3.11.0) 
    -numpy (version 2.5.2) 
    -pandas (version 3.0.5)
    -scikit-learn (version 1.9.0)
    -scipy (version 1.18.0)
## Code Design
It should be noted that for the sake of running the code, the number of simulations should potentially be reduced to 5; during the actual execution, it was 500. It is recommended to have the number of simulations be between 500-1000 for accuracy of results, but because the run time of such a high number is significant, reducing the number is helpful for improving run time.
### Data Loading
The data was imported using pd.read_csv, which loads the provided CSV data file into a two-dimensional Pandas DataFrame, and os.path.join, which combines the file directory into a single path. The features and targets are separated based on their position in the dataset. The last column is the target value, monoclonal antibody concentration. The remaining six columns are the input features, comprised of glucose, glutamine, lactate, ammonia, viable cell density (VCD), and total cell density (TCD), alongside the time in hours (note, time is not a feature). All target and feature columns are then converted to NumPy arrays so that they can be interpreted by scikit-learn to train the regression models. It should be notes that all six features and the mAb concentrations are features of the analysis.
Firstly, 30 batches are selected from the provided datasets outlining various initial conditions. In order to get a representative bank of data to work from, batches were selected randomly and then indexed to ensure that there were no repeated combinations of initial conditions. This is done using an if not function. The results were sorted from lowest to highest and these datasets were then imported into the repository.  
The initial conditions are matched to their respective batch with the isin function, by filtering the dataset of initial conditions and selecting only the batches given by the selected batches list. The data is sorted by the sort_values function based on batch ID and time. It is ensured then that the measurements are ordered chronologically by batch ID.  
### Preprocessing 
First, the data is split into current and lagged sets, based on the lagged (p) value. Say p=2, the first two rows from each batch are not included in the current set, but are instead found in the lagged set. Both VAR and NVAR models use lagged, or past, process measurements to predict future process conditions, while also adding white noise. This is important because the provided data for mAb growth is in the form of a time series, meaning the variables change over time and the values of features at any given point (except for initially) are dependent on the values preceding them. 
The data is then split into training and testing sets. To prevent data leakage between training and test sets, a leave-one-batch-out cross-validation (LOBOCV) method is employed: one batch is left out of the current and lag training sets and is instead used for the testing set. 
### Feature Engineering
The VAR model is linear and thus uses linear regression, which models the relationship between input features and a continuous target variable using a linear equation. For the NVAR models, the polynomial model uses a PolynomialFeatures function. From the pipeline, PolynomialFeatures generates the independent and interactive terms between the input variables, allowing for the capture of combined and non-linear effects in the model. For the knn model, the target value is predicted by locating the distance to the k-nearest neighbouring points, with uniform weighting. The values of the neighbouring points are then used to calculate the expected target.
The data is then standardized using StandardScalarVAR so that it is all on the same scale. This model uses seven independent features with different units; because a k-Nearest Neighbour model is evaluated, this standardization is especially important as the model relies on the distance between points. If the model were not standardized, features with larger magnitudes could have a disproportionate impact on the nearest neighbour estimates.
Firstly, vector autoregression (VAR) was evaluated, with lagged values of 2-3 to determine the appropriate model complexity. Next, non-linear vector autoregression (NVAR) was evaluated, using both polynomial and k-nearest neighbours (knn) as regression model types. The degree or order of the polynomial is varied from 2 to 3 to determine an appropriate model complexity. The number of neighbours was varied from 2 to 5. Lagged values of 1-5 were selected for this regression type.

### Model Training
All of the model types were ranked based on the root-mean-square error (RMSE) average score and standard deviations of both the training and test data. The rankings for the VAR and NVAR assessments are exported to the following respective tables in the table folder: table_02_VAR_summary and table_03_NVAR_summary. 
This was done with different methods, depending on the model type. For VAR, LOBOCV is used—again, it excludes one batch from the current and lagged training sets, and is used for the testing sets. The code then loops through each batch until all batches have been left out once. For the NVAR models, a GridSearchCV function is used, which optimizes the model over a cross-validated grid search. This is also referred to as hyperparameter optimization of the model. The training data is split into three cross-validation folds, with two being used for training and the remaining fold for validation; this is done with the GroupKFold function. Using cross-validation offers a more robust choice than a single train-test split, as the model performance is evaluated over multiple data folds, reducing the metric’s dependence on a single split. GroupKFold is another function that cross-validates data by excluding a batch to prevent data spillage. 
It should be noted that autoregression (AR) was not evaluated. This is because AR only looks at a single variable. VAR and NVAR look at multiple features to learn about the interactions between variables. 
##Forecasting and Optimization
The final portion of the code design is the forecasting and optimization. The VAR and NVAR models use the previous values of the seven features: glucose, glutamine, lactate, ammonia, VCD, TCD, and mAb. VAR uses a linear regression; NVAR uses either polynomial regression or knn to predict the current values, based on the interactions between/likeness of previous features.
To optimize the model, the initial values of glucose, glutamine, and VCD are found; each combination is simulated using the NVAR polynomial degree 2 model. 
The economics of this process are given by the following equations: 

	- Cell culture cost [$/L] = 10 ×(Initial Glucose [mM]) + 20 ×(Initial Glutamine [mM]) + 0.0005 ×(Initial VCD [cells/mL])
	- Cell culture value [$/L] = 1000 ×(mAb [mg/L])
	- Profit [$/L] = (Cell culture value [$/L]) - (Cell culture cost [$/L])
	- Space-time profit [$ / (L ⋅d)] = (Profit [$/L]) / ((Harvest time [d]) + (Downtime [d]))
	- Space-time yield [mg / (L ⋅d)] = (mAb [mg/L]) / ((Harvest time [d]) + (Downtime [d]))

Using these equations, the optimal initial conditions are selected by evaluating the calculated space-time profit and yield. The idxmax function finds the combination of initial conditions that produces the highest values. The 95% prediction intervals are found using 2.5th and 97.5th percentiles of the simulated results.
### Final Model Choice
The final model choice is a second-order NVAR polynomial model. This model is fairly simple, with a relatively small RMSE. The rankings of the mean score and standard deviations are shown in Tables 1 and 2 for the VAR and NVAR assessments, respectively. 
#### Table 1: Vector Autoregression Score Rankings
|p |N simulations |RMSE (mean) [mg/L] |RMSE (SD) [mg/L]   |
|-------------|---|-------------------|-------------------|
|1            |500|0.16507475875526342|0.10327739371904004|
|2            |500|0.15428338265564476|0.09033795210389742|
|3            |500|0.1605487298622014 |0.08082950485026197|
#### Table 2: Non-Linear Vector Autoregression Score Rankings[1^]
|Model|p  |N simulations|RMSE (mean) [mg/L] |RMSE (SD) [mg/L]   |
|-----|---|-------------|-------------------|-------------------|
|poly |1  |5            |0.17343247918114368|0.09351400485922225|
|poly |2  |5            |0.14164955889452946|0.09086647795284529|
|poly |3  |5            |0.14110488449146488|0.09307819832601634|
|poly |4  |5            |0.2821132161349238 |0.26895078046980725|
|poly |5  |5            |0.5991246793452618 |0.15231011663880745|
|knn  |1  |5            |0.17633615822974852|0.10254647959459011|
|knn  |2  |5            |0.17761977830850934|0.10390251955232979|
|knn  |3  |5            |0.19720989249927728|0.11159304684587233|
|knn  |4  |5            |0.22515962583684798|0.12382236862312876|
|knn  |5  |5            |0.21626718079818968|0.10080063725189164|
[1^]Runs of 5 were used to reduce the run time, however for more accurate results, adjusting the number of simulations to 500-1000 is recommended. 

As seen, the best choices for each model type are VAR, p=2, and NVAR with a polynomial model of degree 2, p=2. For VAR, decreasing the complexity does not provide a significant benefit, as the model is simple as is. Furthermore, the error increases. For NVAR, polynomial and knn models provided similar results. Increasing polynomial complexity and nearest neighbours resulted in an increased error. Therefore, for simplicity, the NVAR(2) polynomial model with degree 2 was selected. The forecasted curves for the VAR and NVAR models are shown in Figures 1-9. 
 
![img_1.png](img_1.png)
#### Figure 1: Forecasted VAR(1).
![img_2.png](img_2.png)
#### Figure 2: Forecasted VAR(2).
![img_3.png](img_3.png)
#### Figure 3: Forecasted VAR(3).
For simplicity and comparability, only runs up to p=3 were included for non-linear VAR models. 
![img_6.png](img_6.png)
#### Figure 4: Forecasted NVAR, polynomial model, p=1. 
![img_7.png](img_7.png)
#### Figure 5: Forecasted NVAR, polynomial model, p=2. 
![img_8.png](img_8.png)
#### Figure 6: Forecasted NVAR, polynomial model, p=3. 
![img_9.png](img_9.png)
#### Figure 7: Forecasted NVAR, k-nearest neighbours, p=1.
![img_10.png](img_10.png)
#### Figure 8: Forecasted NVAR, k-nearest neighbours, p=2. 
![img_11.png](img_11.png)
#### Figure 9: Forecasted NVAR, k-nearest neighbours, p=3. 

Polynomial models are simpler and easier to interpret compared to non-parametric models—seeing as the second-order polynomial regression provided a slightly lower root-mean-square error, it is the clear choice for our predictive model.
Another evaluation metric is the white noise of each run. White noise is the difference between predicted and actual values. When plotted, white noise should have a mean close to zero, appear randomly distributed around zero, and show no systematic trends over time. This is important because white noise is sampled directly from the residuals; therefore, it reflects the actual noise within the modelled processes. The further away the white noise is from this criterion, the less representative the simulation becomes. Thus, our final model selection is heavily dependent on white noise. 
The plots for white noise are shown in Figures 10 and 11. 
 

![img_4.png](img_4.png)
#### Figure 10: Plotted white noise, non-linear vector autoregression. From left to right, top to bottom, polynomial degrees 1-5, respectively. 
![img_5.png](img_5.png) 
#### Figure 11: Plotted white noise, non-linear vector autoregression. From left to right, top to bottom, k-Nearest Neighbours, with Neighbours 2-5, respectively. 

### Advantages and Limitations
The primary advantages of this model are its simplicity and interpretability. The model is only a second-order polynomial; however, it is still optimized such that it produces the lowest RMSE, making it the best model fit. Compared to the knn NVAR model, the chosen model is relatively straightforward and easy to understand, while still accounting for quadratic and interactive effects. It also allows for all of the features to be incorporated in the analysis, instead of only one. 
Another advantage of the model is its broad set of data. By selecting random batch runs and ensuring there is no overlap in initial conditions, the model has a greater variety of data to go off of. That being said, in order to get the broadest set of data to train the model, the batches were selected based on uniqueness. This is useful in establishing larger ranges of applicability for the trained model; however, it fails to include any reproducibility. This is important because not all runs with the same initial conditions will produce the same outputs. Having repeated results increases confidence in the data. Because only one data set was used for a given set of conditions, it is possible that the model was trained on an ‘outlier’ run. Including these repeated runs helps to avoid this. 
Another limitation is that only 30 batches were used; if all datasets were used, it would provide a more comprehensive combined dataset; however, for computational efficiency, only 30 batches were used. Furthermore, the model will likely not be able to accurately predict the mAb production outside of the range of given operating conditions, as the model will need to extrapolate beyond its training data. 

### Optimizations and Input Feature Impacts
The optimal results for the space-time profit and yield alongside the 95% prediction intervals are given in Table 1 below. 
#### Table 3: Optimal Space-Time Profit and Yield
|Objective|Glucose [mM]|Glutamine [mM]|VCD [cells/mL]|Harvest Time [d]  |Cell Culture Cost [$/L]|Cell Culture Value Median [$/L]|Cell Culture Value Lower 95% [$/L]|Cell Culture Value Upper 95% [$/L]|Space-Time Profit Median [$/L/d]|Space-Time Profit Lower 95% [$/L/d]|Space-Time Profit Upper 95% [$/L/d]|Space-Time Yield Median [mg/L/d]|Space-Time Yield Lower 95% [mg/L/d]|Space-Time Yield Upper 95% [mg/L/d]|
|---------|------------|--------------|--------------|------------------|-----------------------|-------------------------------|----------------------------------|----------------------------------|--------------------------------|-----------------------------------|-----------------------------------|--------------------------------|-----------------------------------|-----------------------------------|
|profit   |20.0        |1.0           |200000.0      |2.6666666666666665|320.0                  |605.1694501837226              |444.913146180798                  |775.1793889255417                 |77.77348641374255               |34.06722168567217                  |124.13983334332958                 |0.16504621368646982             |0.12133994895839946                |0.21141256061605684                |
|yield    |25.0        |4.0           |1200000.0     |1.3333333333333333|930.0                  |504.8485859224478              |359.34175835087837                |651.7700940410402                 |-182.20774889037955             |-244.5678178496236                 |-119.24138826812565                |0.21636367968104908             |0.15400361072180502                |0.27933004030330294                |

As seen, optimal conditions vary based on the maximized variable. We can see that the maximum space-time profit has relatively low concentrations for glucose, glutamine, and VCD, initially. On the other hand, the maximum space-time yield has higher initial concentrations.
Based on these results, we can see that our profit and yield optima differ; specifically, we can see that the 95% yield interval for the space-time profit is negative. We can conclude that by maximizing production efficiency, we are not necessarily maximizing the economics of the process.
To maximize the space-time profit, the following initial conditions are recommended: a glucose concentration of 20 mM, a glutamine concentration of 1 mM, and a VCD of 200,000 cells/mL. Harvest time is recommended to be 2.67 days. Here, cell cost is $320/L, median cell culture value at harvest is $605.17/L ; the 95% prediction interval is $444.91-775.18/L. The median space-time profit is then $77.77/(L*d), with its 95% prediction interval being $34.07-124.14/(L*d). The estimated median mAb space-time yield is 0.1650 mg/(L·d), with its 95% prediction interval being 0.1213-0.2114 mg/(L·d).

##Applications
This model can be useful in determining the optimal conditions for monoclonal antibody growth without the need to run experiments. Furthermore, the model can be useful in providing estimates of cell growth for changes in glucose, glutamine, lactate, ammonia, VCD, and TCD. It is also helpful in comparing production efficiency and economic performance. 
