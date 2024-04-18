import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Iris veri setini yükleme
iris = datasets.load_iris()

# Veri setini bir DataFrame'e dönüştürme
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target'])

# 'target' sütununu 'species' adıyla değiştirme ve sınıf isimlerini hedef değişken değerlerine dönüştürme
data['species'] = iris.target_names[iris.target]

# 'species' sütununu sayısal bir formata dönüştürme
species_to_num = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
data['species_encoded'] = data['species'].map(species_to_num)

# Veri setinin ilk beş gözlemine bakma
print("Veri setinin ilk beş gözlemi:")
print(data.head())

#---
# Veri setinin genel bilgileri
#---

print("\nVeri setinin genel bilgileri:")
print(data.info())

#---
# Veri setinin betimlenmesi
#---

print("\nVeri setinin istatistiksel özeti:")
print(data.describe())

#---
# Eksik veri kontrolü
#---

print("\nEksik veri kontrolü:")
print(data.isnull().sum())

#---
# Aykırı değer analizi
#---

# Sepal uzunlukları için kutu grafiği
sns.boxplot(x='species', y='sepal length (cm)', data=data)
plt.title('Sepal Uzunluklarına Göre Iris Türleri')
plt.show()

# Petal uzunlukları için kutu grafiği
sns.boxplot(x='species', y='petal length (cm)', data=data)
plt.title('Petal Uzunluklarına Göre Iris Türleri')
plt.show()

#---
# Veri dağılımlarının görselleştirilmesi
#---

# Sepal uzunlukları için histogram
sns.histplot(data=data, x='sepal length (cm)', hue='species', kde=True)
plt.title('Sepal Uzunluklarına Göre Iris Türleri')
plt.show()

# Petal uzunlukları için histogram
sns.histplot(data=data, x='petal length (cm)', hue='species', kde=True)
plt.title('Petal Uzunluklarına Göre Iris Türleri')
plt.show()

#---
# Korelasyon analizi
#---

# 'species' ve 'target' sütunlarını veri setinden çıkarma
numerical_data = data.drop(['target', 'species', 'species_encoded'], axis=1)

correlation_matrix = numerical_data.corr()
print("\nKorelasyon matrisi:")
print(correlation_matrix)

# Korelasyon matrisinin görselleştirilmesi
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()

#---
# Veri setinin eğitim ve test alt kümelerine bölünmesi
#---

X = numerical_data
y = data['species_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nEğitim verisi boyutu:", X_train.shape)
print("Test verisi boyutu:", X_test.shape)
