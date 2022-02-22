# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Oliver created the model. It is random forest classifier using the hyperparameters declared in params.yaml.

## Intended Use

This model should be used to predict the salary based off a handful of attributes. The attributes are declared in data.py. This is the place to activate or inactivate the features to be part of the model.

## Training Data

The original data was obtained from uci and cleaned (spaces remove, duplicates dropped, unkonwn countries removed).
Then the cleaned data was split in a training and test dataset (20%). This can be configured in params.yaml. The training data
was again split into two parts (training, validation dataset). Again the validation size is 20% and could be configured in params.yaml.

## Evaluation Data

The model was evaluated by applying the test data with confirmation of 0,81 precision.

## Metrics

The model was evaluated having a precision of 0.80 with the validation set and 0,81 with test data.
Other scores are available and stored as model metrics (./model/score.json).

## Ethical Considerations

Bias and fairness was not part of this project. But you can see with the data slice performance test, that one should put more emphasize on it.

## Caveats and Recommendations

This project re. model was NOT focused on model performance, but on modelops and its CI/CD. For further model improvements all is set and the model parameters and features can be easily adapated (e.g. params.yaml, data.py).
