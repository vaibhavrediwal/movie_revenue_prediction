# Movie Revenue Prediction Project

## Project Overview
This project aims to predict worldwide box office revenue for movies released between 2000 and 2009 using machine learning techniques. It implements both Random Forest and Deep Learning models to analyze various features of movie releases and make revenue predictions.

## Data
The dataset contains information on 2000 movie releases from 2000 to 2009. Key features include:
- Release Group
- Worldwide revenue
- Domestic revenue and percentage
- Foreign revenue and percentage
- Release year

## Models
1. Random Forest Regressor
2. Deep Learning Neural Network (using TensorFlow/Keras)

## Procedure
1. Data Preprocessing:
   - Clean and format currency and percentage columns
   - Handle missing or infinite values
2. Feature Engineering:
   - Calculate profit margin
   - Create a binary feature for summer releases
3. Exploratory Data Analysis:
   - Visualize worldwide box office trends
4. Model Training:
   - Split data into training and testing sets
   - Scale features using StandardScaler
   - Train Random Forest and Deep Learning models
5. Model Evaluation:
   - Calculate Mean Squared Error and R-squared scores
   - Perform cross-validation for Random Forest
6. Visualization:
   - Plot model performance and feature importance

## Results and Conclusion
- Random Forest Model:
  - Demonstrates excellent performance with an R-squared score of 0.9922
  - Mean Cross-validation Score: 0.9423
  - Reliable for predicting movie revenues

- Deep Learning Model:
  - Currently underperforming with a negative R-squared score
  - Requires further optimization and tuning

The Random Forest model shows high accuracy in predicting movie revenues, explaining 99.22% of the variance in the target variable. Its consistent performance across cross-validation suggests robustness.

The Deep Learning model's poor performance indicates a need for significant improvements, potentially in architecture, hyperparameters, or training process.

Overall, the project successfully implements a predictive model for movie revenues, with the Random Forest algorithm proving to be particularly effective for this dataset.
