import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from tabulate import tabulate
import matplotlib.pyplot as plt

X = pd.read_csv("X_test.csv").astype('float')
Y = pd.read_csv("Y_test.csv").astype('float')
Y=np.array(Y)
Y=Y.ravel()

print("1	Normal")
print("2	Ischemic changes (Coronary Artery)")
print("3	Old Anterior Myocardial Infarction")
print("4	Old Inferior Myocardial Infarction")
print("5	Sinus tachycardia")
print("6	Sinus bradycardia")
print("7	Ventricular Premature Contraction (PVC)")
print("8	Supraventricular Premature Contraction")
print("9	Left bundle branch block")
print("10	Right bundle branch block")
print("11	Left ventricle hypertrophy")
print("12	Atrial Fibrillation or Flutter")
print("13	Others")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print ("...........................................................................")
print ("Select any of the following Machine Learning Algorithm from the list below.")
print ("1. Support Vector Machine")
print ("2. Logistic Regression")
print ("3. K-Nearest Neighbours")
print ("4. Naive Bayes")
print ("5. Compare results")
print ("...........................................................................")


################################################## TAKING INPUT HERE #################

n = input("Enter your number here: ")
if n!= 5:
    row = input("Enter a row number for ECG custom input: ")

def plot(r):
    x = list(range(1,188))
    y = r
    plt.plot(x, y)
    plt.show()


def run():
    print()
    print("..................................Traing set................................")
    print()
    clf.fit(X_train, Y_train)
    print(clf.predict(X_train))
    score = clf.score(X_train, Y_train)
    print ("Training Set accuracy = ", score*100)
    print ("Training Set error = ", (1-score)*100)

    print()
    print("..................................Test set...................................")
    print()
    print(clf.predict(X_test))
    score = clf.score(X_test, Y_test)
    print ("Test Set accuracy = ", score*100)
    print ("Test Set error = ",(1-score)*100)

def predict_custom_input(row):
    print("---------------------------------------")
    custom_predict = clf.predict([X.iloc[row]])
    plot(X.iloc[row])
    if custom_predict == 0:
        print ("Normal")
    elif custom_predict == 1:
        print ("Ischemic changes (Coronary Artery)")
    elif custom_predict == 2:
        print ("Old Anterior Myocardial Infarction")
    elif custom_predict == 3:
        print ("Old Inferior Myocardial Infarction")
    elif custom_predict == 4:
        print ("Sinus tachycardia")
    elif custom_predict == 5:
        print ("Sinus bradycardia")
    elif custom_predict == 6:
        print ("Ventricular Premature Contraction (PVC)")
    elif custom_predict == 7:
        print ("Supraventricular Premature Contraction")
    elif custom_predict == 8:
        print ("Left bundle branch block")
    elif custom_predict == 9:
        print ("Right bundle branch block")
    elif custom_predict == 10:
        print ("Left ventricle hypertrophy")
    elif custom_predict == 11:
        print ("Atrial Fibrillation or Flutter")
    elif custom_predict == 12:
        print ("Others")
    else:
        print("Out of Bound.")

def result_comparison():
    clf_svm = svm.SVC(C=1,kernel="linear")
    clf_lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf_knn = KNeighborsClassifier(n_neighbors=5)
    clf_nb = MultinomialNB()

    clf_svm.fit(X_train, Y_train)
    clf_lr.fit(X_train, Y_train)
    clf_knn.fit(X_train, Y_train)
    clf_nb.fit(X_train, Y_train)

    table = [["SVM", clf_svm.score(X_train, Y_train), clf_svm.score(X_test, Y_test)],["LR", clf_lr.score(X_train, Y_train), clf_lr.score(X_test, Y_test)],["KNN", clf_knn.score(X_train, Y_train), clf_knn.score(X_test, Y_test)],["NB", clf_nb.score(X_train, Y_train), clf_nb.score(X_test, Y_test)]]
    print tabulate(table, headers=["Classifier","Training set accuracy", "Testing set accuracy"])

    # objects = ('SVM', 'LR', 'KNN', 'NB')
    # y_pos = np.arange(len(objects))
    # performance = [1.0, 0.8, 0.6, 0.4]
    
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.ylabel('Accuracy')
    # plt.title('Comparison of different Machine Learning Classifiers for detecting ECG based heart diseases')
    # plt.show()

    n_groups = 4
    means_frank = (clf_svm.score(X_train, Y_train), clf_lr.score(X_train, Y_train), clf_knn.score(X_train, Y_train), clf_nb.score(X_train, Y_train))
    means_guido = (clf_svm.score(X_test, Y_test), clf_lr.score(X_test, Y_test), clf_knn.score(X_test, Y_test), clf_nb.score(X_test, Y_test))
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    
    rects1 = plt.bar(index, means_frank, bar_width,
    alpha=opacity,
    color='b',
    label='Training set accuracy')
    
    rects2 = plt.bar(index + bar_width, means_guido, bar_width,
    alpha=opacity,
    color='g',
    label='Testing set accuracy')
    
    plt.xlabel('Person')
    plt.ylabel('Scores')
    plt.title('Comparison of different Machine Learning Classifiers for detecting ECG based heart diseases')
    plt.xticks(index + bar_width, ('SVM', 'LR', 'KNN', 'NB'))
    plt.legend()
    
    plt.tight_layout()
    plt.show()


if n == 1:
    clf = svm.SVC(C=1,kernel="linear")
    print ("Running Support Vector machine")
    run()
    predict_custom_input(row)

elif n == 2:
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    print ("Running Logistic Regression")
    run()
    predict_custom_input(row)
    
elif n == 3:
    clf = KNeighborsClassifier(n_neighbors=5)
    print ("Running KNN")
    run()
    predict_custom_input(row)
 
elif n == 4:
    # clf = GaussianNB()
    clf = MultinomialNB()
    print ("Running Naive Bayes")
    run()
    predict_custom_input(row)

elif n == 5:
    print ("comparing results")
    result_comparison()

else:
    print("Wrong number selected")


