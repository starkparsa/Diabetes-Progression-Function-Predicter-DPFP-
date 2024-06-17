# Diabetes-Progression-Function-Predicter-DPFP

## Problem Statement
The Diabetes Pedigree Function (DPF) is a tool used to assess genetic risk for diabetes based on family history, introduced in 1993 as part of the Diabetes Genetics Initiative. It calculates a score indicating an individual's likelihood of developing diabetes, which increases with each affected first-degree relative. This score helps categorize individuals into risk groups, though genetics is just one aspect of diabetes risk alongside lifestyle and environmental factors.

## Data Analysis and Modeling Approach
### Data Exploration:

The dataset used is the Pima Indians Diabetes Database, containing 768 entries and 9 columns including features like Glucose, Blood Pressure, BMI, and the target variable, DiabetesPedigreeFunction (DPF).
### Data Pre-processing:

 - Zeros in critical columns (Glucose, BloodPressure, etc.) were replaced with NaNs to represent missing data more accurately.
 - Simple Imputer was used with the median strategy to fill in missing values, ensuring data integrity for analysis.
### Modeling:

 - Random Forest Regression: Implemented with 50 estimators, achieving a Mean Absolute Error (MAE) of approximately 0.254.
 - XGBoost Regression: Utilized with a learning rate of 0.009 and max leaves of 10, resulting in a lower MAE of about 0.242.
### Conclusion:

 - XGBoost outperformed Random Forest in predicting the DPF based on the dataset, indicating its suitability for this specific problem. Further refinement and optimization could enhance predictive accuracy, leveraging both genetic and clinical data for diabetes risk assessment.
 - This approach integrates data exploration, pre-processing, and machine learning modeling to derive insights and predictive capabilities relevant to diabetes risk assessment using the Diabetes Pedigree Function.
