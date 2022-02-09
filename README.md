## Code structure
* `main.py` is to train a new model
* `generate_test.py`: use the best stored model in order to generate the final CSV file to be submitted.
* `config.ini` has the configurations in general
* `logs` to store the log data
  * `model_performance.json` contains all the trained models + the id for the best model
* `utilities.py` is a set of supporting functions
* `models` contain the stored model
* `output` generated predictions using the best model
* 
## Overview on the task
To understand what a FEDAS code is, I checked [this document](https://www.sgidho.com/SiteAssets/SitePages/Guideline%20FEDAS%20Product%20Classification%20KEY%20-%20PCK/I-3%20FEDAS%20Introduction%20EN%20V4.0%20www%2015.11-2018.pdf), which basically states that it is actually 6 numbers:
1   -> product type
2-3 -> activity code
4-5 -> product main group
6   -> product subgroup

Thus, I think this number should be broken down to these elements, and a multi-label classification/regression problem is to be performed.

Since there is no meaning for the order here (55 is not smaller than 56), then one-hot encoding is the 

## Limitations and TODO
* I used label encoding when I should have used one-hot encoding, but the matrix bloat, and my computer can't handle it
* Only RF model can be trained now
* The logging doesn't store the training configurations (selected features, pre-processing steps, and model hyper-parameters)
* No hyper-parameters search and selection is performed
* Max performance on cross-validation is `58%` only
  * Cleaning and investing time in the features would definitely get better results
  * Proper encoding