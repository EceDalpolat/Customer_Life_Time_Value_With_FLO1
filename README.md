# Customer_Life_Time_Value_With_FLO1

# CLTV Prediction with BG-NBD and Gamma-Gamma
# Proje Özeti
Bu proje, FLO'nun satış ve pazarlama faaliyetleri için yol haritası oluşturmasına yardımcı olmak amacıyla, müşteri yaşam boyu değerini (CLTV) tahmin etmeyi hedeflemektedir. Bu tahminler, müşterilerin geçmiş alışveriş davranışlarına dayanarak yapılmış ve BG-NBD ve Gamma-Gamma modelleri ile desteklenmiştir.

#Veri Seti
Veri seti, FLO müşterilerinin 2020 ve 2021 yıllarında hem online hem de offline alışveriş yaptıkları OmniChannel alışveriş verilerinden oluşmaktadır. Müşterilerin toplam alışveriş sayıları, alışveriş yaptıkları kanallar ve toplam harcamalar gibi özellikler içerir.

Değişkenler
master_id: Eşsiz müşteri numarası
order_channel: Alışveriş yapılan platformun adı (Android, iOS, Desktop, Mobile, Offline)
last_order_channel: En son alışverişin yapıldığı kanal
first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
last_order_date: Müşterinin yaptığı son alışveriş tarihi
last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
order_num_total_ever_offline: Müşterinin offline platformda yaptığı toplam alışveriş sayısı
customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# Kurulum ve Çalıştırma
Bu projeyi çalıştırmak için aşağıdaki adımları izleyebilirsiniz:
Gerekli Python kütüphanelerini yükleyin:

pip install pandas lifetimes scikit-learn
Projeyi çalıştırmak için kodu indirin ve CSV dosyasını ilgili dizine yerleştirin:

git clone <repository_url>
cd CLTVPrediction

Veriyi işleyin ve CLTV tahminini oluşturun:
python cltv_prediction.py
# Görevler
Görev 1: Veriyi Hazırlama
Veri seti yüklenip, aykırı değerler belirlendi ve baskılandı.
Müşterilerin toplam alışveriş sayısı ve toplam harcamaları için yeni değişkenler oluşturuldu.
Tarih değişkenleri uygun formata dönüştürüldü.
Görev 2: CLTV Veri Yapısının Oluşturulması
Analiz tarihi, veri setindeki son alışveriş tarihinden 2 gün sonrasına ayarlandı.
Müşterilerin haftalık bazda recency, T, frequency, ve monetary değerleri hesaplandı.
Görev 3: BG/NBD ve Gamma-Gamma Modellerinin Kurulması ve CLTV Hesaplanması
BG/NBD modeli kullanılarak, müşterilerin 3 ve 6 ay içindeki satın alma tahminleri yapıldı.
Gamma-Gamma modeli kullanılarak, müşterilerin bırakacakları ortalama değer tahmin edildi.
6 aylık CLTV hesaplanarak veriye eklendi ve en yüksek CLTV'ye sahip 20 müşteri gözlemlendi.
Görev 4: Segmentlerin Oluşturulması ve Aksiyon Önerileri
CLTV değerlerine göre müşteriler 4 segmente ayrıldı (A, B, C, D).
Her bir segment için aksiyon planı önerildi.
Aksiyon Önerileri
Segment A:
VIP Müşterilere Özel Kampanyalar: Bu segment, şirkete en yüksek CLTV değerini sunan müşterileri içerir. Bu nedenle, özel indirimler ve sadakat programları önerilebilir.
Kişiselleştirilmiş Ürün Önerileri: Müşterilere ilgilendikleri kategorilere göre kişisel alışveriş önerileri sunulabilir.
Segment D:
Aktivasyon Kampanyaları: Bu müşterilerle yeniden etkileşime geçmek için e-posta ve bildirim kampanyaları düzenlenebilir.
Fiyat Avantajı ve Promosyonlar: Daha düşük fiyat avantajı sunarak, müşterilerin alışveriş sıklığını artırmaya yönelik stratejiler önerilebilir.
Dosyalar
cltv_prediction.py: Tüm veri işleme ve CLTV hesaplama sürecinin yer aldığı Python dosyası.
cltv_prediction.csv: Model sonuçlarının yer aldığı çıktı dosyası.
Yazar
Bu proje, FLO için müşteri yaşam boyu değerini tahmin etmek amacıyla geliştirilmiştir.

