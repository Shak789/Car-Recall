## Overview
This project uses logistic regression to predict vehicle recall risk based on National Highway Traffic Safety Administration (NHTSA) consumer complaint data. It can also be extended to identify vehicles at risk of recall before official action is taken by the NHTSA, as well as identify features that lead to recalls.

## Data Sources
[NHTSA Datasets](https://www.nhtsa.gov/nhtsa-datasets-and-apis): Complaint and recall information. Complaints are aggregated by vehicle make, model and year and merged with recall data. A vehicle is considered recalled if NHTSA has issued a recall for that specific make, model and year at least once.
Since the complaint dataset was not merged with recalls by vehicle component, over 85% of complaint profiles are matched to recalls. While predicting recalls based on vehicle components would decrease the imbalance, it also significantly reduces the sample size per group since complaints span many components. This would reduce the signal from features needed to predict recall.

## Features
1.	**KeyBERT Safety Score**: KeyBERT was used to find keywords from vehicle complaints matched to recalls. Each vehicle make was then scored by the overlap of complaints in the description. TF-IDF was first used, but the keyword list provided was not relevant, since TF-IDF does not consider the semantic meaning of words. 
The score achieved a clear boundary between recalled and non-recalled vehicles. Recalled vehicles produced a median score of 23, which was nearly three times higher than the median score of 9 in non-recalled vehicles.
2.	**First Year Complaint Proportion:** Measures the vehicle's first-year complaints against the manufacturer’s total first-year complaints. This metric is normalized to isolate early defect patterns regardless of overall sales volume.
3.	**Median mileage:** Median vehicle mileage at time of complaint.

The model was first trained and tested on 2010-2020 data.

**Pivoting from Classification to Probabilistic Outputs**
The project focuses on probabilistic outputs to predict recalls rather than binary classification. Classification uses an arbitrary threshold to separate between recalled and non-recalled vehicles, but it does not illustrate the differences between the likelihood of vehicles being recalled. For example, a vehicle with a 55% probability of recall should not be viewed the same as a vehicle which has a 95% probability of recall. Using probabilities allows regulators and manufacturers to distinguish between ambiguous defect signals and high-confidence safety anomalies. 

**Comparing Models**
Three models (logistic regression, random forest, XGBoost) were trained and tested on data from 2010 to 2020. Since a probabilistic approach is prioritized, it is important to verify if the models are calibrated.

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Calibration.png)

The diagrams show that each of the models are uncalibrated and underpredicting the probability of recall when class weighting is used. However, if class weighting is removed, the model will start only predicting the majority class, which renders the probabilities meaningless. 

Platt scaling was applied to correct the uncalibrated curves, but it pulled predicted probabilities toward the base recall rate, rather than producing a well-calibrated curve.

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Scaling.png)

This demonstrates the interpretation should shift from absolute probabilities to a ranking model with “risk scores” on a scale from 0 to 100. Therefore, vehicles are sorted according to scores, enabling regulators and manufacturers to view vehicles with high risk of recall.

Given the shift to ranking, ROC-AUC scoring was used to compare models. The AUC of all models after hyperparameter tuning are listed below: 
1.	Logistic Regression: 0.717
2.	Random Forest: 0.689
3.	XGBoost: 0.681
   
Since there were small differences in AUC, bootstrap confidence intervals of AUC were computed for each of the models. Logistic regression maintained a statistically significant AUC advantage over XGBoost (95% bootstrap CI on the difference: [0.004, 0.054], excluding 0). The advantage over random forest was smaller and not statistically significant the 95% level (CI: [-0.004, 0.079]).

Given the intervals are very small, all models were evaluated on whether they could segment vehicle risk scores into priority tiers to ensure the ranking is meaningful.

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Segment.png)

This chart shows that the risk score for logistic regression is monotonically increasing for every bracket, compared to random forest which has inconsistent changes in recall rate across the tiers. Therefore, logistic regression was chosen as the final model since it can provide meaningful rankings for risk scores and manufacturers and regulators can find value in the model’s outputs. 

**Testing on Unseen Data**
The logistic regression model was tested on data from 2022 to 2026. The AUC score decreased to 0.61. 

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Segment2.png)

The model’s diminishing performance may be explained by the following:
1.	Early warning of vehicle recalls: The model is flagging some vehicles as high risk that may not yet have received official recall. There were 31 non-recalled vehicles (approximately 15% of non-recalled vehicles) with risk scores above 70, which explains the decrease in recall rate for vehicles with a risk score of 70-80 and 80-90. This is useful for manufacturers to proactively investigate vehicles rather than wait for NHTSA action. Even if the vehicles are not recalled by NHTSA, the high scores indicate the vehicles have problematic complaints that may reduce sales.
2.	Complaint accumulation lag: KeyBERT complaint scores decrease for recent years as newer vehicles have not accumulated enough complaints yet, particularly 2025-2026.
3.	The model struggles with predicting risk scores for near-luxury, luxury, or truck models which barely meet complaint volume thresholds and whose owners often bypass NHTSA reporting. Luxury owners often visit dealers before filing formal complaints, while truck owners may route issues through maintenance channels.  Future iterations of the model could monitor warranty claims and dealer service visit volumes since NHTSA data alone will not flag these risks.

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Cumulative.png)

The cumulative distribution shows that most unrecalled cars are clustering at 0-50 which means the model is correctly giving most unrecalled cars lower scores. 
The distribution of scores leads to 3 distinct tiers for evaluating recalls:
1.	Low Risk (0 - 49): Continue Routine Monitoring
2.	Medium Risk (50 - 69): Monitor Vehicle Closely
3.	High Risk (70 - 100): Investigate Immediately

**SHAP Analysis**

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/SHAP.png)

The SHAP analysis shows the feature impact is balanced. Normalizing complaint levels was a reasonable choice since complaint volume dominated in previous iterations. The SHAP values also demonstrates that low mileage failures are more concerning than high mileage ones. 

**Conclusion**

This project provides manufacturers and regulators with an interpretable model for predicting vehicle recall risk. Logistic regression was selected as the final model through a structured process. After applying class weighting and hyperparameter tuning to all model, bootstrap confidence intervals demonstrated that logistic regression had a statistically significant AUC advantage over XGBoost, and the risk-tier analysis showed more consistent, monotonic recall rates than random forest. Combined with the SHAP analysis, manufacturers and regulators can use a meaningful and explainable model, rather than an arbitrarily chosen algorithm.
Beyond prediction, the model offers practical value. It can be used as an early warning system by flagging vehicles before official NHTSA action, and it distinguishes between "sudden" and "slow burn" recall patterns, which require different monitoring strategies.


