# deep-learning-challenge
Dependencies include:
- train_test_split from sklearn.model_selection
- StandardScaler from sklearn.preprocessing 
- tensorflow as tf
- pandas as pd

# Overview 
Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using the dataset, a binary classifier model is created to predict whether applicants will be successful if funded by Alphabet Soup.

The dataset is a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset the following columns:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

Original model was created using the 'deep-learning-challenge' notebook. Each optimization attempt was saved in a different numbered notebook. The Kerasturner Search was conducted in Optimization4 and the best model was created and saved to the AlphabetSoupCharity_Optimized h5 file.

# Results
## Data Preprocessing
Data was read in from a csv file to a pandas dataframe. Data was cleaned by dropping columns, and creating dummies for categorical fields using pd.get_dummies(). Data was split using train_test_split(). Training and test data was then scaled using StandardScaler().

The 'IS_SUCCESSFUL' variable was used as the target for each model.
EIN and NAME were removed from the dataset, and were not targets or features.
All other variables were used as features in the other models. One model (nn_optimazation3) did not use APPLICATION_TYPE or CLASSIFICATION.

## Compiling, Training, and Evaluating the Model
The Original Model was run with 2 layers (layer 1: 80 nodes, layer 2: 30 nodes), with a relu activation. This model was saved as an h5 file (AlphabetSoupCharity.h5) in this repository. With a goal of .75 accuracy, changes to the model were attempted to increase accuracy. A summary of the results of those models is below:

### Original Model-
*   Loss: 0.559
*   Accuracy: 0.731

### nn_optimization1-
*   Loss: 0.556
*   Accuracy: 0.729
*   Changes: Classification Bins were increased
*   No significant change to accuracy or loss

### nn_optimization2-
*   Loss: 0.556
*   Accuracy: 0.729
*   Changes: A Hidden layer was added, and neurons were added
*   No significant change to accuracy or loss

### nn_optimization3-
*   Loss: 0.602
*   Accuracy: 0.701
*   Changes: the data being evaluated removed the CLASSIFICATION and APPLICATION_TYPE data fields
*   Model appears to be worse. Loss increased and Accuracy decreased.

After three attempts to improve accuracy of the model failed to do so, a kerasturner search was run to find the best hyperparameters. The kerasturner seach used 'relu', 'tanh' and 'sigmoid' activations, 1-51 neurons in each layer (using a step of 5), and 1-6 layers. It was run with a max epoch of 50, and 2 hyperband iterations. The best model this search found were:
- Activation: sigmoid
- Number of layers: 5
- Number of Nuerons: 1(46), 2(31), 3(11), 4(46), 5(21)
- Loss: .573
- Accuracy: .733

Creating this model and running it through 100 epochs:
- Loss: .553
- Accuracy: .730

# Summary
Using the above models on the data that was provided, a model that could produce .75 could not be found. Eliminating Classification and Application Type lowered the accuracy and increased loss, indicating that these variables are important to the model. Further experiementing could be done eliminating other combinations of variables. Also, the maximum number of epochs run was 100. In the models used, it appeard that improvement leveled off around 50 epochs. With a different dataset, this may change, and more epochs could help produce a better model. Given the best model from the kerasturner seach had 5 layers, a future model should consider using more layers (4-6).