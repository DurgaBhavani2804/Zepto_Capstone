# Titanic Project

This project uses the Titanic dataset to first explore the data, then build machine learning models to predict who survived.

## What's in this folder

- **01_eda.ipynb** — This is the first notebook. It loads the data, cleans it, and explores it with charts (Part A of the task).
- **02_modeling.ipynb** — This is the second notebook. It builds the machine learning models (Part B of the task).
- **titanic.csv** — The dataset saved as a file, so we don't have to download it again.
- **full_pipeline.joblib** — The final trained model saved so it can be reused later.
- **charts** folder — All the pictures/graphs saved from the notebooks.

## How to run it

1. Open the folder in VS Code.
2. Run `01_eda.ipynb` from top to bottom first. This one needs internet the first time because it downloads the data.
3. Then run `02_modeling.ipynb` from top to bottom. This one uses the saved `titanic.csv`, so it doesn't need internet.

## What I found in Part A (the data exploration)

- Some columns had missing data. `deck` was missing about 77% of the time, so I just dropped it, too much was missing to guess. `age` was missing about 20% of the time, so I filled the gaps with the median age. `embarked` was missing very little, so I just dropped those few rows.
- `age` had 65 outliers and `fare` had 114 outliers (checked using the IQR method).
- `fare` is right-skewed, meaning most people paid a small amount but a few people paid a lot.
- Women survived a lot more than men (about 74% vs 19%).
- Richer passengers (1st class) survived more than poorer ones (3rd class).
- The biggest correlations were between `pclass` and `fare`, and between `sibsp` and `parch` (makes sense, families travel together).
- Overall, the people most likely to survive were: women, kids, people in 1st/2nd class, and people traveling in small families (not alone, not in huge families).

## What I found in Part B (the models)

- I split the data 80% train / 20% test, keeping the survive/not-survive ratio the same in both (stratified split).
- I tried 3 models: Logistic Regression, Decision Tree, and Random Forest.
- **Random Forest performed the best overall** — about 81.6% accuracy and the best F1 score.
- Logistic Regression was close behind and actually had the best AUC score.
- Decision Tree was good at precision but missed a lot of actual survivors (low recall).
- I also tried fixing the class imbalance (more people didn't survive than did) using `class_weight` and SMOTE — both helped a bit, about the same amount.
- I tuned the Random Forest using GridSearchCV, but it didn't actually beat the simple default version on the test data.
- I also did a bonus regression task: predicting `fare` from the other columns. It wasn't very accurate (R² around 0.4), and the errors get bigger for expensive tickets (heteroscedasticity).

## Final pick

I'd go with the **Random Forest** model. It had the best balance of accuracy, F1, and recall, so it's the most reliable at actually catching survivors correctly.

## Using the saved model

```python
import joblib
import pandas as pd

model = joblib.load("full_pipeline.joblib")

new_passenger = pd.DataFrame([{
    "pclass": 3, "sex": "male", "age": 22.0,
    "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"
}])

print(model.predict(new_passenger))
```


