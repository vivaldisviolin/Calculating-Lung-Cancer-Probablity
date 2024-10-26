from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# Iris veri setini yükle
lung_cancer = pd.read_csv("lung cancer data.csv")
# 'gender' sütununu 0 ve 1 olarak değiştirin
lung_cancer['GENDER'] = lung_cancer['GENDER'].map({'M': 0, 'F': 1})



# Özellikler ve hedef değişkenleri ayır
x = lung_cancer.drop(columns = ["LUNG_CANCER"])  # 'target' sütununu hedef değişken olarak kabul ediyoruz
y = lung_cancer["LUNG_CANCER"]

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Karar ağacı sınıflandırıcısını oluştur ve eğit
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
tree_clf.fit(x_train, y_train)

# Kullanıcıdan veri al
gender = int(input("Cinsiyetinizi giriniz(Erkek için 0, Kadin için 1): "))
age = int(input("Yaşınızı girin: "))
smoking = int(input("Sigara içiyor musunuz? (Evet için 1, Hayır için 0): "))
yellow_fingers = int(input("Sari tirnak hastaliginiz var mi(Evet için 1, Hayır için 0): "))
anxiety = int(input("Anksiyete var mı? (Evet için 1, Hayır için 0): "))
peer_pressure = int(input("Akran baskısı var mı? (Evet için 1, Hayır için 0): "))
chronic_disease = int(input("Kronik hastalık var mı? (Evet için 1, Hayır için 0): "))
fatigue = int(input("Yorgunluk var mı? (Evet için 1, Hayır için 0): "))
allergy = int(input("Alerji var mı? (Evet için 1, Hayır için 0): "))
wheezing = int(input("Hırıltı var mı? (Evet için 1, Hayır için 0): "))
alcohol_consumption = int(input("Alkol tüketimi var mı? (Evet için 1, Hayır için 0): "))
coughing = int(input("Öksürük var mı? (Evet için 1, Hayır için 0): "))
shortness_of_breath = int(input("Nefes darlığı var mı? (Evet için 1, Hayır için 0): "))
swallowing_difficulty = int(input("Yutma zorluğu var mı? (Evet için 1, Hayır için 0): "))
chest_pain = int(input("Göğüs ağrısı var mı? (Evet için 1, Hayır için 0): "))


# Kullanıcıdan alınan verileri bir listeye koy
user_data = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consumption, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]

feature_importances = tree_clf.feature_importances_

value = 0

for i in range(15):
    if user_data[i] == 1:
        value = value + feature_importances[i]
print(f"Kanser olma ihtimaliniz: {value*100}")

'''
# Test seti üzerinde tahmin yap
y_pred = tree_clf.predict(x_test)

# Doğruluk skorunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk Skoru:", accuracy)

# Karışıklık matrisini hesapla
cnf_mtrx = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:\n", cnf_mtrx)

# Karar ağacını görselleştir
plt.figure(figsize=(15,10))
plot_tree(tree_clf, filled=True, feature_names = x.columns, class_names = ["Class 0", "Class 1"])
plt.show()



feature_importances = tree_clf.feature_importances_
feature_names = x.columns

features_importances_sorted = sorted(zip(feature_importances,feature_names))
    

for importance,name in features_importances_sorted:
    print(f"{name}: {round(importance*100,2)}")
'''



