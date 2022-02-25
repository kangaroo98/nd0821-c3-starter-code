# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Oliver created the model. It is a random forest classifier using the hyperparameters declared in params.yaml:
train_model:
random_state: 42
test_size: 0.2
param_grid:
n_estimators: [20, 60]
max_features: ["auto", "sqrt"]
max_depth: [4, 5, 10]
criterion: ["gini", "entropy"]

## Intended Use

This model should be used to predict the salary as a target based on a handful of features. The features are defined in ml/data.py and are divided in categorical and numeric features (lists cat_features and num_features). You can activate or inactivate a feature by putting it in comment in the lists. You have to reflect this in the pydantic model in the same manner (main.py)

## Training Data

The original data was obtained from uci and is cleaned by running clean_data.py (spaces removed, duplicates dropped, unkonwn countries removed). The cleaned data is stored in data/cleaned_census.csv.
The cleaned data is then split in a training and test dataset (20%). This can be configured in params.yaml. The training data is again split into two parts (training, validation dataset). The current validation size is 20% and could be configured in params.yaml as well.

## Evaluation Data

The model was evaluated by applying the test data with a confirmation of the trained model precision. In addition the performance of all categorical features were evaluated via data slicing.

## Metrics

The model was evaluated having a precision of 0.82, a recall of 0.52 and a fbeta 0.64. The result was validated with the test dataset and confirmed the model metrics with a precision of 0.82, a recall of 0.55 and a fbeta 0.66.
The scores are available and stored as model metrics (./model/score.json). In addition the slicing data is stored in /model/slice_output.txt as requested.

## Ethical Considerations

Bias and fairness was so far not evaluated in detail. But the data slice performance test showed, that a deeper analysis might be necessary e.g. regarding specific countries.

## Caveats and Recommendations

This project re. model was not focused on model performance, but on modelops and its CI/CD. For further model improvements all is set and the model parameters and features can be easily adapated (e.g. params.yaml, data.py).
