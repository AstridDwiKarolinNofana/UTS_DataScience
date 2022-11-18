import numpy as np
from dataset import class_dataset
from missing_value import class_missing_value
from classifier import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

if __name__ == '__main__':

    "Memanggil Dataset RandomForestClassifier"

    data = class_dataset(dataset_name="heart_disease_patients")

    "preprocesing/normalisasi"
    

    """
    model RandomForestClassifier with kfold
    """
    data.y = np.array([data.y[1] for i in data.y])
    kf = KFold(n_splits=3, random_state=None, shuffle=False)
    acc = []
    for train_index, test_index in kf.split(data.X):
      X_train, X_test = data.X[train_index], data.X[test_index]
      y_train, y_test = data.y[train_index], data.y[test_index]
      RandomForestClassifier = class_RandomForestClassifier(X_train, y_train)
      RandomForestClassifier.model()
      DecisionTreeClassifier = class_DecisionTreeClassifier(X_train, y_train)
      DecisionTreeClassifier.model()
      y_pred = RandomForestClassifier.predict(X_test)
      y_prediksi = DecisionTreeClassifier.predict(X_test)
      #acc.append(accuracy_score(y_test, y_pred))
      acc = accuracy_score(y_test, y_pred)
      acc1 = accuracy_score(y_test, y_prediksi)
      print("accuracy RandomForestClassifier : ", acc)
      print("accuracy DecisionTreeClassifier : ", acc1)
 



    # """
    # model RandomForestClassifier
    # """
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data.X, data.y, test_size=0.17, random_state=40)
    # RandomForestClassifier = class_RandomForestClassifier(X_train, y_train)
    # RandomForestClassifier.model()
    # y_pred = RandomForestClassifier.predict(X_test)

    # """
    # evaluasi
    # """
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy : ", acc)    

        

    # """
    # model DecisionTreeClassifier
    # """
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data.X, data.y, test_size=0.17, random_state=42)
    # DecisionTreeClassifier = class_DecisionTreeClassifier(X_train, y_train)
    # DecisionTreeClassifier.model()
    # y_pred = DecisionTreeClassifier.predict(X_test)

    # """
    # evaluasi
    # """
    # acc = accuracy_score(y_test, y_pred)
    # print("accuracy : ", acc)



    #"""
    #model SVM
    #"""
    #X_train, X_test, y_train, y_test = train_test_split(
    #    data.X, data.y, test_size=0.2, random_state=40)
    #svm = class_SVM(X_train, y_train)
    #svm.model()
    #y_pred = (np.rint(svm.predict(X_test))).astype(int)

    #"""
    #evaluasi
    #"""
    #acc = accuracy_score(y_test, y_pred)
    #print("accuracy : ", acc)






