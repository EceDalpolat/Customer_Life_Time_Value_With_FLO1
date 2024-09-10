##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
df_=pd.read_csv("FLOCLTVPrediction/flo_data_20k.csv")
df=df_.copy()
df.head()
df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]
df["order_num_total"]=df["order_num_total_ever_offline"]+df["order_num_total_ever_online"]
df.describe().T
df.isnull().sum()
df=df[(df["customer_value_total"]>0)]
df.dropna(inplace=True)
def outlier_thresholds(dataframe,col_name):
    quartil1=dataframe[col_name].quantile(0.25)
    quartil3=dataframe[col_name].quantile(0.75)
    interquantile_range=quartil3-quartil1
    up_limit=quartil3+1.5*interquantile_range
    low_limit=quartil1-1.5*interquantile_range
    return round(low_limit),round(up_limit)

def replace_with_thresholds(dataframe,variable):
    low_limit,up_limit=outlier_thresholds(dataframe,variable)
    dataframe.loc[dataframe[variable]<low_limit,variable]=low_limit
    dataframe.loc[dataframe[variable]>up_limit,variable]=up_limit
columns_to_check = ["order_num_total_ever_online",
                    "order_num_total_ever_offline",
                    "customer_value_total_ever_offline",
                    "customer_value_total_ever_online"]
for col in columns_to_check:
    replace_with_thresholds(df,col)

date_columns=df.columns[df.columns.str.contains("date")].tolist()
df[date_columns] = df[date_columns].apply(pd.to_datetime)
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
analysis_date=df["last_order_date"].max()+pd.DateOffset(days=2)
cltv_df=pd.DataFrame()
cltv_df["customer_id"]=df["master_id"]
cltv_df["recency_cltv_weekly"]=((df["last_order_date"]-df["first_order_date"]).dt.days)/7
cltv_df["T_weekly"]=((analysis_date-df["first_order_date"]).dt.days)/7
cltv_df["frequency"]=df["order_num_total"]
cltv_df["monetary_cltv_avg"]=(df["customer_value_total"]/cltv_df["frequency"])
cltv_df.head()
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
bgf=BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(cltv_df["frequency"],cltv_df["recency_cltv_weekly"],cltv_df["T_weekly"])
cltv_df["exp_sales_3_months"]=bgf.predict(12,cltv_df["frequency"],cltv_df["recency_cltv_weekly"],cltv_df["T_weekly"])
cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])


ggf = GammaGammaFitter()
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["cltv"] = ggf.customer_lifetime_value(
    bgf,
    cltv_df['frequency'],
    cltv_df['recency_cltv_weekly'],
    cltv_df['T_weekly'],
    cltv_df['monetary_cltv_avg'],
    time=6,  # 6 aylık CLTV hesaplama
    freq="W",  # Verinin haftalık olduğu belirtiliyor
    discount_rate=0.01  # İndirgeme oranı
)

# CLTV'si en yüksek 20 müşteri
top_20_customers = cltv_df.sort_values(by="cltv", ascending=False).head(20)
# En yüksek CLTV değerine sahip 20 müşteri
top_20_customers[["customer_id", "cltv"]]
cltv_df.head()
cltv_final=cltv_df.copy()
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
cltv_final["segment"]=pd.qcut(cltv_final["cltv"], 4, labels=["D","C","B","A"])
cltv_final.head()
cltv_final.sort_values(by="cltv", ascending=False).head(50)
cltv_final.dtypes
numerical_cols = cltv_final.select_dtypes(include=['number']).columns
cltv_final.groupby("segment")[numerical_cols].agg({"sum","mean","count"})
# BONUS: Tüm süreci fonksiyonlaştırınız.

def create_cltv(df,month=3):
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df["order_num_total"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df = df[(df["customer_value_total"] > 0)]
    df.dropna(inplace=True)

    def outlier_thresholds(dataframe, col_name):
        quartil1 = dataframe[col_name].quantile(0.25)
        quartil3 = dataframe[col_name].quantile(0.75)
        interquantile_range = quartil3 - quartil1
        up_limit = quartil3 + 1.5 * interquantile_range
        low_limit = quartil1 - 1.5 * interquantile_range
        return round(low_limit), round(up_limit)

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
        dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

    columns_to_check = ["order_num_total_ever_online",
                        "order_num_total_ever_offline",
                        "customer_value_total_ever_offline",
                        "customer_value_total_ever_online"]
    for col in columns_to_check:
        replace_with_thresholds(df, col)

    date_columns = df.columns[df.columns.str.contains("date")].tolist()
    df[date_columns] = df[date_columns].apply(pd.to_datetime)
    analysis_date = df["last_order_date"].max() + pd.DateOffset(days=2)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = df["master_id"]
    cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7
    cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).dt.days) / 7
    cltv_df["frequency"] = df["order_num_total"]
    cltv_df["monetary_cltv_avg"] = (df["customer_value_total"] / cltv_df["frequency"])
    cltv_df.head()
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])
    cltv_df["exp_sales_3_months"] = bgf.predict(12, cltv_df["frequency"], cltv_df["recency_cltv_weekly"],
                                                cltv_df["T_weekly"])
    cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    ggf = GammaGammaFitter()
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])
    cltv_df["cltv"] = ggf.customer_lifetime_value(
        bgf,
        cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'],
        cltv_df['monetary_cltv_avg'],
        time=month,  # 6 aylık CLTV hesaplama
        freq="W",  # Verinin haftalık olduğu belirtiliyor
        discount_rate=0.01  # İndirgeme oranı
    )
    top_20_customers = cltv_df.sort_values(by="cltv", ascending=False).head(20)
    # En yüksek CLTV değerine sahip 20 müşteri
    top_20_customers[["customer_id", "cltv"]]
    cltv_df.head()
    cltv_final = cltv_df.copy()
    # GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
    # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
    # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
    cltv_final["segment"] = pd.qcut(cltv_final["cltv"], 4, labels=["D", "C", "B", "A"])
    cltv_final.head()
    cltv_final.sort_values(by="cltv", ascending=False).head(50)
    cltv_final.dtypes
    numerical_cols = cltv_final.select_dtypes(include=['number']).columns
    cltv_final.groupby("segment")[numerical_cols].agg({"sum", "mean", "count"})

    return  cltv_final
cltv_analiz_csv=create_cltv(df,month=6)
cltv_analiz_csv.to_csv("cltv_prediction.csv")
