# Veri Analizi ve Çoklu Doğrusal Regresyon Projesi

Bu proje, veri analizi ve çoklu doğrusal regresyon yöntemlerini uygulamak için Python kullanarak gerçekleştirilmiştir. Proje kapsamında, iki farklı veri seti kullanılmıştır: birincisi rastgele üretilen bir veri seti ve ikincisi ise klasik Iris veri setidir.

## Rastgele Veri Seti
Rastgele veri seti, projenin temel veri kaynağını oluşturur. Bu veri seti, belirli bir fonksiyon kullanılarak oluşturulan ve ardından çeşitli veri analizi teknikleriyle incelenen bir yapay veri setidir.

### Veri Oluşturma
Rastgele veri seti oluşturulurken, normal dağılımlı rastgele değişkenler kullanılmıştır. Bu değişkenler, belirli bir fonksiyonla birbirlerine bağlıdır ve buna göre bağımlı değişken yaratılmıştır.

### Veri Dönüşümü
Veri setinde bulunan faktörel ve sayısal değişkenler belirlenmiş ve ilgili dönüşümler gerçekleştirilmiştir. Faktörel değişken kategorik bir değişkene dönüştürülürken, sayısal değişken doğrudan sayısal olarak korunmuştur.

### Eksik Veri Tespiti ve Doldurma
Eksik verilerin tespiti yapılmış ve eksik veriler ortalama değerler ile doldurulmuştur. Ayrıca, aykırı değerler de görselleştirilmiştir.

### Veri Dağılımlarının Görselleştirilmesi
Histogramlar, kutu grafikleri ve Q-Q plotlar aracılığıyla veri dağılımları görselleştirilmiştir.

### Korelasyon Analizi
Veri setindeki değişkenler arasındaki korelasyonlar incelenmiş ve korelasyon katsayıları hesaplanmıştır.

### Çoklu Doğrusal Regresyon ve Multicollinearity
Çoklu doğrusal regresyon modeli oluşturulmuş ve model özeti incelenmiştir. Ayrıca, multicollinearity durumu da korelasyon matrisi görselleştirilerek değerlendirilmiştir.

### Veri Bölme
Veri seti, eğitim ve test alt kümelerine bölünmüştür. Bu bölünme, modelin doğruluğunu değerlendirmek için kullanılmıştır.

## Iris Veri Seti

Iris veri seti, projenin ikinci veri kaynağını oluşturur. Bu veri seti, ünlü Iris bitkisinin çiçek özelliklerini içeren ve çeşitli sınıflara ayrılmış bir veri setidir.

### Veri Yükleme ve Dönüştürme
Iris veri seti, scikit-learn kütüphanesinden yüklenmiş ve daha sonra bir DataFrame'e dönüştürülmüştür.

### Veri Betimlenmesi
Veri setinin istatistiksel özeti incelenmiş ve genel bilgiler elde edilmiştir.

### Eksik Veri Kontrolü
Veri setinde eksik veri olup olmadığı kontrol edilmiştir.

### Aykırı Değer Analizi
Sepal ve petal uzunluklarına göre iris türlerinin kutu grafikleri çizilmiştir.

### Veri Dağılımlarının Görselleştirilmesi
Sepal ve petal uzunlukları için histogramlar çizilmiş ve iris türlerine göre dağılımlar incelenmiştir.

### Korelasyon Analizi
Veri setindeki değişkenler arasındaki korelasyonlar incelenmiş ve korelasyon matrisi çıkarılmıştır.

### Veri Setinin Eğitim ve Test Alt Kümelerine Bölünmesi
Veri seti, eğitim ve test alt kümelerine bölünmüştür. Bu bölünme, modelin doğruluğunu değerlendirmek için kullanılmıştır.

## Kullanılan Kütüphaneler ve Araçlar

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Statsmodels

Bu README dosyası, proje hakkında genel bir bilgilendirme sağlamak amacıyla hazırlanmıştır. Daha detaylı bilgi almak için ilgili kod dosyalarını inceleyebilirsiniz.
