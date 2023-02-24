#-------------------------------------------------------------------------
# AUTHOR: Michael Acosta
# FILENAME: decision_tree_2.py
# SPECIFICATION:    Create a decision tree from a set of features. 
#                   Trains the tree and test the performance with accuracy.
# FOR: CS 4210- Assignment #2
# TIME SPENT: About 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    Age = {
        "Young": 1,
        "Presbyopic": 2,
        "Prepresbyopic": 3,
    }
    SpectaclePrescription = {
        "Myope": 1,
        "Hypermetrope": 2,
    }
    Astigmatism = {
        "No": 1,
        "Yes": 2,
    }
    TearProductionRate = {
        "Reduced": 1,
        "Normal": 2,
    }
    X = dbTraining.copy() #make a copy of file read in
    for i, row in enumerate(dbTraining): 
        #transform features based on dictionary
        X[i] = [Age[row[0]], SpectaclePrescription[row[1]], 
                Astigmatism[row[2]], TearProductionRate[row[3]]]

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    RecommendedLenses = {
        "No": 1,
        "Yes": 2,
    }
    Y = [0] * len(dbTraining) #create empty list
    for i, row in enumerate(dbTraining):
        Y[i] = RecommendedLenses[row[4]] #transform label based on dictionary

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)

        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #transform features based on dictionary
            data = [Age[data[0]], SpectaclePrescription[data[1]], 
                    Astigmatism[data[2]], TearProductionRate[data[3]], RecommendedLenses[data[4]]]
            class_predicted = clf.predict([[data[0], data[1], data[2], data[3]]])[0]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if class_predicted == data[4]:
                if data[4] == 2:
                    tp += 1
                if data[4] == 1:
                    tn += 1
            else:
                if data[4] == 2:
                    fn += 1
                if data[4] == 1:
                    fp += 1

    #find the average of this model during the 10 runs (training and test set)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("final accuracy when training on " + ds + ": " + str(accuracy))