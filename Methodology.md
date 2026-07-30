## Overview
This project uses logistic regression to predict vehicle recall risk based on National Highway Traffic Safety Administration (NHTSA) consumer complaint data. It can also be extended to identify vehicles at risk of recall before official action is taken by the NHTSA, as well as determine which features lead to recalls.

The full project code can be found [here](https://github.com/Shak789/Car-Recall/blob/master/Car%20Recall%20Notebook.ipynb).

## Data Sources
[NHTSA Datasets](https://www.nhtsa.gov/nhtsa-datasets-and-apis): Complaint and recall information. Complaints are aggregated by vehicle make, model and year and merged with recall data. A vehicle is considered recalled if NHTSA has issued a recall for that specific make, model and year at least once.
Since the complaint dataset was not merged with recalls by vehicle component, over 85% of complaint profiles are matched to recalls. While predicting recalls based on vehicle components would decrease the imbalance, it also significantly reduces the sample size per group since complaints span many components. This would reduce the signal from features needed to predict recall.

## Features
1.	**KeyBERT Safety Score**: KeyBERT was used to find keywords (on training data) from vehicle complaints matched to recalls. KeyBERT is only performed once and each vehicle is then scored by the overlap of complaints in the description. This eliminates the need to run KeyBERT on every incoming complaint during the full pipeline. TF-IDF was first used, but the keyword list provided was not relevant, since TF-IDF does not consider the semantic meaning of words.
   
The score achieved a clear boundary between recalled and non-recalled vehicles. Recalled vehicles produced a median score of 23, which was nearly 2.5x times higher than the median score of 9 in non-recalled vehicles.

2.	**First Year Complaint Proportion:** Measures a vehicle model's first-year complaints relative to the manufacturer’s total first-year complaints. The ratio uses stabilizing constants in the numerator and denominator to reduce noise from vehicles with low-volume complaints.
   
3.	**Median mileage:** Median vehicle mileage at time of complaint.

The model was first trained and tested on 2010-2020 data.

## Pivoting from Classification to Probabilistic Outputs
The project focuses on probabilistic outputs to predict recalls rather than binary classification. Classification uses an arbitrary threshold to separate between recalled and non-recalled vehicles, but it does not illustrate the differences between the likelihood of vehicles being recalled. For example, a vehicle with a 55% probability of recall should not be viewed the same as a vehicle which has a 95% probability of recall. Using probabilities allows regulators and manufacturers to distinguish between ambiguous defect signals and high-confidence safety anomalies. 

## Comparing Models
Three models (logistic regression, random forest, XGBoost) were trained and tested on data from 2010 to 2020. Since a probabilistic approach is prioritized, it is important to verify if the models are calibrated.

The diagrams show that each of the models are uncalibrated and underpredicting the probability of recall when class weighting is used. The Brier scores of approximately 0.2 indicate a root-mean-squared probability error of approximately 45% relative to the actual outcomes. However, if class weighting is removed, the model will collapse into predicting the majority class, which renders the probabilities meaningless.

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Images/Calibration.png)

While Platt scaling and isotonic regression corrected the uncalibrated curves, they pulled predicted probabilities toward the recall rate of the entire dataset (approximate mean probabilities of 0.85). This demonstrates that we cannot discriminate between the probabilities for recalling vehicles.

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Images/Platt_Scaling_Isotonic_Regression.png)

Due to the uncalibrated probabilities, the interpretation should shift from absolute probabilities to a ranking model with "risk scores" on a scale from 0 to 100. Vehicles are sorted according to scores, enabling regulators and manufacturers to view vehicles with high risk of recall.

Given the shift to ranking, ROC-AUC scoring was used to compare models. The AUC of all models after hyperparameter tuning are listed below: 
1. Logistic Regression: 0.731
2. Random Forest: 0.745
3. XGBoost: 0.741

Since there were small differences in AUC, 95% confidence intervals (CIs) for both individual AUC scores and pairwise AUC differences were computed using bootstrapping with 2,000 resamples. Each model showed overlapping AUC confidence intervals and all pairwise difference intervals (LR–RF, LR–XGB, and RF–XGB) captured 0, In addition, the absolute values of upper limits and lower limits of the pairwise difference intervals remained below 0.05. This means that there is no statistically significant difference in the AUC between the three models.

The models were evaluated on whether they could segment vehicle risk scores into priority tiers to ensure the ranking is meaningful.

The chart below shows that the risk score for logistic regression is monotonically increasing for every bracket, compared to random forest which has inconsistent changes in recall rate across the tiers. Therefore, logistic regression was chosen as the final model since it can provide meaningful rankings for risk scores and manufacturers and regulators can find value in the model’s outputs.

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Images/Segmentation_Final.png)


## Testing on Unseen Data
The logistic regression model was tested on data from 2022 to 2026. The AUC score decreased to 0.63 on 2022-2026 data. 

The model’s diminishing AUC can be explained by the following:

1.	Early warning of vehicle recalls: The model is flagging some vehicles as high risk that may not yet have received official recall. There were over 25 non-recalled vehicles (approximately 15% of non-recalled vehicles) with risk scores above 70. This is shown by the slight plateau in recall rate for vehicles with a risk score of 70-80 in the diagram below. The early warning system is useful for manufacturers to proactively investigate vehicles rather than wait for NHTSA action. Even if NHTSA does not issue a recall, the high risk scores show that the vehicles have many problematic complaints that can affect consumer trust and sales.

   ![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Iamges/Segment2.png)
   
2.	Complaint accumulation lag: KeyBERT complaint scores decrease for recent years as newer vehicles have not accumulated enough complaints yet, particularly 2025-2026.

## Model Limitations
The model struggles with predicting risk scores for vehicles where the complaint volume narrowly meets thresholds (less than 15 complaints), illustrating that owners may bypass NHTSA reporting. The vehicles can be divided into two specific segments:
   
1. Luxury/Near-Luxury (e.g. Audi, BMW, Mercedes-Benz, Lincoln): Since luxury buyers expect premium customer care, they likely visit dealerships before escalating issues to the NHTSA.

2. Commercial Fleet (e.g. Chevrolet Silverado, Ford F-150): Owners may route mechanical issues through institutional maintenance channels. These channels will resolve defects significantly faster than using the NHTSA portal since periods without vehicle use can lead to compounding revenue loss.

The model's struggles in predicting risk scores are further illustrated by the following histograms. Both histograms show that the model has assigned higher risk scores to recalled vehicles than non-recalled vehicles and is successfully ranking risk. However, there is an overlap between 40 and 60, showing the model is struggling in discriminating complaints for some vehicle segments (luxury/near-luxury, commercial fleet).
   
![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Images/Distribution.png)

The model's primary value still lies in ranking recall risk rather than predicting recalls with absolute certainty. Regulators and manufacturers do not necessarily need perfectly calibrated predictions. Instead, they need the model to rank higher-risk vehicles above lower-risk ones, which is confirmed by the distributions and recall rate increasing across tiers.

The distribution of scores leads to 3 distinct tiers for evaluating recalls:
1.	Low Risk (0 - 49): Continue Routine Monitoring
2.	Medium Risk (50 - 69): Monitor Vehicle Closely
3.	High Risk (70 - 100): Investigate Immediately

## SHAP Analysis

![Formula](https://raw.githubusercontent.com/shak789/Car-Recall/master/Images/shap_xgb_dot.png)

The SHAP analysis shows three important insights:
1. Normalizing the first-year complaint proportion by manufacturer volume was a reasonable decision. In earlier iterations, complaint volume drowned out the other features, but the graph above shows balanced SHAP values.
   
2. The median mileage feature behaves as expected. Low mileage at time of complaint is more indicative of recall than high mileage. Failures at low mileage suggest early manufacturing defects, which are more likely to lead a recall, compared t0 high mileage failures that are more consistent with normal wear and tear.

3. The KeyBERT score contributes meaningfully along with the other features. The model is able to find patterns in complaint language that match serious safety defects.

## Conclusion
This project provides manufacturers and regulators with an interpretable model for predicting vehicle recall risk. Logistic regression was selected as the final model through a structured process. After applying class weighting and hyperparameter tuning to all models, bootstrap confidence intervals demonstrated that there was no statistically significant difference in the AUC between logistic regression, random forest and XGBoost models. The risk-tier analysis showed more consistent, monotonic recall rates than random forest and XGBoost. Combined with the SHAP analysis, manufacturers and regulators can use a meaningful and explainable model. Beyond prediction, the model offers practical value. It can be used as an early warning system by flagging vehicles before official NHTSA action and the limitations are shown through the complaint patterns of luxury/near-luxury and commercial fleet vehicle owners.
