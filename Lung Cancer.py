#Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset
lung_cancer = pd.read_csv("lung cancer data.csv")

#Changing "0" & "1" integers insted of "M" & "F" stings.
lung_cancer['GENDER'] = lung_cancer['GENDER'].map({'M': 0, 'F': 1})



#Defining target column and dropping it
x = lung_cancer.drop(columns = ["LUNG_CANCER"]) 
y = lung_cancer["LUNG_CANCER"]

#Spliting dataset for test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Setting Decision tree classifier features & Training model 
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
tree_clf.fit(x_train, y_train)

#Collecting information about target person
gender = int(input("Enter your gender (0 for Male, 1 for Female): "))
age = int(input("Enter your age: "))
smoking = int(input("Do you smoke? (1 for Yes, 0 for No): "))
yellow_fingers = int(input("Do you have yellow nail syndrome? (1 for Yes, 0 for No): "))
anxiety = int(input("Do you have anxiety? (1 for Yes, 0 for No): "))
peer_pressure = int(input("Is there peer pressure? (1 for Yes, 0 for No): "))
chronic_disease = int(input("Do you have a chronic disease? (1 for Yes, 0 for No): "))
fatigue = int(input("Do you experience fatigue? (1 for Yes, 0 for No): "))
allergy = int(input("Do you have allergies? (1 for Yes, 0 for No): "))
wheezing = int(input("Do you experience wheezing? (1 for Yes, 0 for No): "))
alcohol_consumption = int(input("Do you consume alcohol? (1 for Yes, 0 for No): "))
coughing = int(input("Do you have a cough? (1 for Yes, 0 for No): "))
shortness_of_breath = int(input("Do you experience shortness of breath? (1 for Yes, 0 for No): "))
swallowing_difficulty = int(input("Do you have difficulty swallowing? (1 for Yes, 0 for No): "))
chest_pain = int(input("Do you have chest pain? (1 for Yes, 0 for No): "))


user_data = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consumption, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]

#Defining features importances coefficient
feature_importances = tree_clf.feature_importances_
value = 0

#Calculating Probability
for i in range(15):
    if user_data[i] == 1:
        value = value + feature_importances[i]
print(f"Probability of cancer: {value*100}")

'''

y_pred = tree_clf.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accurucy Score:", accuracy)


cnf_mtrx = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cnf_mtrx)


plt.figure(figsize=(15,10))
plot_tree(tree_clf, filled=True, feature_names = x.columns, class_names = ["Class 0", "Class 1"])
plt.show()



feature_importances = tree_clf.feature_importances_
feature_names = x.columns

features_importances_sorted = sorted(zip(feature_importances,feature_names))
    

for importance,name in features_importances_sorted:
    print(f"{name}: {round(importance*100,2)}")
'''



