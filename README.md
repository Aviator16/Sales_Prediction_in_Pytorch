# Sales_Prediction_in_Pytorch
Predicting sales of items in stores using Feed Forward Neural Network, Long Short Term Memory, Temporal Convolution Network &amp; a hybrid of TCN and LSTM models.

![sales](https://github.com/Aviator16/Sales_Prediction_in_Pytorch/blob/main/sales.jpg?raw=true)

The task requires predicting sales for a given item from a given store on a future date. The dataset consists of Date, Store, Item and Sale attributes. Out of the 4 features, 2 are categorical variable, 1 is date variable and 1 is continious which is our target. The two categorical variables, store and item, consists of 10 unique stores and 50 unique items. The date variable ranges from the beginning of 2013 to the end of 2017.

* I split the data into training set, which has all the data from the years 2013 to 2016, and test set which consists of data from 2017. 
* The training data has 7,30,500 samples and the test data has 1,82,500 samples.
* For validation a leave-6-out strategy has been used wherein 6 months is used as validation set and the rest 42 months is trained on.
* The first 6 months and then the last 6 months of every year is used validation in each batch giving us 8 batches for 4 years. For eg. in the first batch of training, the first 6 months of 2013 is used for validation, and the rest of the 42 months are used for training; in the next batch the last 6 months of 2013 is used for validation and it continues like that for all the years in training set.

I split the date column to **year**, **month (1-12)**,**day-of-week (1-7)** & **day-of-month (1-31)**. The total number of variables after preprocessing would result in more then 115 features and the datset consists of 9,13,000 samples so I used **Categorical Embeddings** for feature selection to reduce the complexity of the model. It resulted in 60 unique variables.

I have used several models for a comparative study and have done it using **PyTorch**. For each of these models, I have used **Mean Square Error** as my loss function and have used **Adam optimiser** for the purpose of training my weights. I have trained the models on a Ryzen 5 Hexa core 4600H CPU.
### 1. Feed Forward Neural Network:
It consists of 3 hidden layers having 512, 128 and 32 nodes each. Each batch ran for 32 epochs meaning in total the model trained for 256 epochs. Evaluating the model on the test set gave the following results:
* R2 score on test set is 0.923525640363284
* Mean Absolute Error on test set is 6.846392510761627
* Root Mean Square error on test set is 8.725558357325898
* Mean Absolute Percentage Error on test set is 0.15158092560938577
* Adjusted R2 score on test set is 0.9235231260413219
### 2. Long Short Term Memory Neural Network:
It consists of 2 LSTM layers and 2 Fully connected layers and an output layer having 112,96,64 & 16 nodes respectively. Each batch ran for 15 epochs meaning in total the model trained for 120 epochs. Evaluating the model on the test set gave the following results:
* R2 score on test set is 0.9130831700643351
* Mean Absolute Error on test set is 7.067161381278626
* Root Mean Square error on test set is 9.302233612700073
* Mean Absolute Percentage Error on test set is 0.15067687205231767
* Adjusted R2 score on test set is 0.9130803124151123
### 3. Temporal Convolution Network:
It consists of 2 TCN layers and 2 Fully connected layers and an output layer having 112,96,64 & 16 nodes respectively. Each batch ran for 50 epochs meaning in total the model trained for 400 epochs. Evaluating the model on the test set gave the following results:
* R2 score on test set is 0.22736186203334408
* Mean Absolute Error on test set is 20.519154600854115
* Root Mean Square error on test set is 27.734693681675626
* Mean Absolute Percentage Error on test set is 0.41503353019106237
* Adjusted R2 score on test set is 0.22733645925719492
### 4. Hybrid Network of TCN & LSTM:
It consists of 1 TCN layer, 1 LSTM layer and 2 Fully connected layers and an output layer having 112,96,64 & 16 nodes respectively. Each batch ran for 10 epochs meaning in total the model trained for 80 epochs. Evaluating the model on the test set gave the following results:
* R2 score on test set is -0.0698741653697148
* Mean Absolute Error on test set is 25.29713485937249
* Root Mean Square error on test set is 32.63635535486093
* Mean Absolute Percentage Error on test set is 0.5577620565618376
* Adjusted R2 score on test set is -0.06990934066406718

## Conclusion
