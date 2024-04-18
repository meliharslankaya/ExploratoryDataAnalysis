import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split

#---
# Veri setinin oluşturulması
#---

# Rastgele veri oluşturma için seed belirleme
np.random.seed(123)  

# Gözlem sayısı
n = 100  

# Bağımsız değişkenlerin oluşturulması
x1 = np.random.normal(10, 2, n)  
x2 = np.random.normal(5, 1, n)   
x3 = np.random.normal(3, 0.5, n) 
x4 = np.random.normal(7, 1.5, n) 
x5 = np.random.normal(15, 3, n)  

# Bağımlı değişkenin oluşturulması
y = 3 + 2*x1 - 1.5*x2 + 0.5*x3 + 1.8*x4 - 1.2*x5 + np.random.normal(0, 2, n)  

# Veri setinin bir DataFrame'e dönüştürülmesi
data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

# Oluşturulan veri setinin gösterilmesi
print("Oluşturulan veri seti:")
print(data.head())
#---

#---
# Veri dönüşümü
#---

# Faktörel ve sayısal değişkenlerin oluşturulması
x1_f = np.random.normal(10, 2, n)
x2_f = np.random.normal(5, 1, n)

# Faktörel değişken dönüşümü
x1_f_factor = pd.Series(x1_f).astype('category')

# Sayısal değişken dönüşümü
x2_f_numeric = pd.Series(x2_f)

# Değişken türlerinin kontrol edilmesi
print("\nFaktörel değişkenin türü:", x1_f_factor.dtype)
print("Sayısal değişkenin türü:", x2_f_numeric.dtype)
#---

#---
# Eksik veri tespiti ve doldurma
#---

# Eksik veri tespiti
is_na_x1 = pd.isna(x1)
is_na_x2 = pd.isna(x2)

# Eksik verilerin ortalama ile doldurulması
mean_x1 = np.mean(x1)
x1_filled = np.where(is_na_x1, mean_x1, x1)

# Aykırı değerlerin görselleştirilmesi
print("\nAykırı değerlerin görselleştirilmesi:")
data.boxplot()
plt.show()
#---

#---
# Veri dağılımlarının görselleştirilmesi
#---

print("\nHistogram:")
data['x1'].hist()
plt.show()

print("\nKutu grafiği:")
data['x2'].plot(kind='box')
plt.show()

print("\nQ-Q plot:")
stats.probplot(data['x3'], dist="norm", plot=plt)
plt.show()
#---

#---
# Korelasyon Analizi
#---

# Korelasyon katsayısının hesaplanması
correlation = data[['x1', 'y']].corr().iloc[0,1]

# Korelasyon katsayısının gösterilmesi
print("\nKorelasyon katsayısı:", correlation)
#---

#---
# Çoklu Doğrusal Regresyon ve Multicollinearity
#---

# Çoklu doğrusal regresyon modelinin oluşturulması
X = data[['x1', 'x2', 'x3']]
X = sm.add_constant(X)
y = data['y']
model = sm.OLS(y, X).fit()

# Model özetinin gösterilmesi
print("\nÇoklu doğrusal regresyon modeli özeti:")
print(model.summary())

# Korelasyon matrisinin görselleştirilmesi
print("\nKorelasyon matrisi görselleştirmesi:")
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
#---

#---
# Veri Bölme
#---

# Veri setinin eğitim ve test alt kümelerine bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Veri boyutlarının gösterilmesi
print("\nEğitim verisi boyutu:", X_train.shape)
print("Test verisi boyutu:", X_test.shape)
#---
