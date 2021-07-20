# Sales_Prediction_in_Pytorch
Predicting sales of items in stores using Feed Forward Neural Network, Long Short Term Memory, Temporal Convolution Network &amp; a hybrid of TCN and LSTM models.

![sales](https://github.com/Aviator16/Sales_Prediction_in_Pytorch/blob/main/sales.jpg?raw=true)

The task requires predicting sales for a given item from a given store on a future date. The dataset consists of Date, Store, Item and Sale attributes. Out of the 4 features, 2 are categorical variable, 1 is date variable and 1 is continious which is our target. The two categorical variables, store and item, consists of 10 unique stores and 50 unique items. The date varibale ranges from the beginning of 2013 to the end of 2017.

I split the data into training set, which has all the data from the years 2013 to 2016, and test set which consists of data from 2017. The training data has 7,30,500 samples and the test data has 1,82,500 samples. For validation a leave-6-out strategy has been used wherein 6 months is used as validation set and the rest 42 months is trained on. The first 6 months and then the last 6 months of every year is used validation in each batch giving us 8 batches for 4 years. For eg. in the first batch of training, the first 6 months of 2013 is used for validation, and the rest is used for training; in the next batch the last 6 months of 2013 is used for validation and it continues like that for all the years in training set.
