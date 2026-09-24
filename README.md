# 🏦 Garanti BBVA BES: going back to a competition I thought I'd done well in

**EN —** In May 2024 I joined the Garanti BBVA Data Day case study on Kaggle. The task was to guess
how much extra money each customer would put into their private pension (BES) in December 2018, on
top of their regular payments. Scoring was RMSE. I finished **11th out of 51**, and for a long time I
was pretty happy with that.

Two years on, I opened the notebook again and wanted to know how good the model really was. The honest
answer turned out to be: not very. When I test it the way the task actually works, it does worse than
simply predicting the same number for every customer. This repo is me working through why, one
mistake at a time, and then building the model again properly.

**TR —** Mayıs 2024'te Kaggle'daki Garanti BBVA Data Day vakasına katıldım. Görev, her müşterinin
Aralık 2018'de bireysel emeklilik (BES) hesabına düzenli ödemesinin üstüne ne kadar ek para
yatıracağını tahmin etmekti. Puanlama RMSE ileydi. **51 kişi içinde 11.** oldum ve uzun süre bundan
gayet memnundum.

İki yıl sonra notebook'u tekrar açtım ve modelin gerçekte ne kadar iyi olduğunu merak ettim. Cevap pek
iç açıcı değildi. Görevin gerçekte işlediği şekilde test edince, model herkese aynı sayıyı söylemekten
bile kötü çıkıyor. Bu repo, bunun nedenlerini tek tek bulmaya çalıştığım ve sonra modeli düzgünce
yeniden kurduğum yer.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Tweedie-green)
![Tests](https://img.shields.io/badge/tests-pytest-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📊 Where things stand / Nereden nereye

**EN —** I scored every version the same way: to predict a month, the model only gets to see the
months before it, just like the real task (train up to November, predict December). I made all my
choices by looking at June to October, and kept November aside as a final check I didn't tune on.
Next to each score I also put how it compares to the laziest possible model, which predicts the
average for everyone. On this data that turns out to be a surprisingly hard number to beat.

**TR —** Her sürümü aynı şekilde puanladım: bir ayı tahmin ederken model sadece ondan önceki ayları
görebiliyor, tıpkı gerçek görevdeki gibi (Kasım'a kadar eğit, Aralık'ı tahmin et). Bütün kararlarımı
Haziran–Ekim'e bakarak verdim, Kasım'ı ise hiç ayar yapmadığım son bir kontrol olarak kenarda tuttum.
Her skorun yanına bir de akla gelebilecek en tembel modelle, yani herkese ortalamayı söyleyen modelle
karşılaştırmayı koydum. Bu veride o sayıyı geçmek sandığımdan çok daha zor çıktı.

| # | Version | Jun–Oct RMSE | vs mean | Months it beats the mean (of 5) | Nov RMSE | vs mean |
|---|---------|:-------:|:-------:|:---:|:-------:|:-------:|
| 0 | Just predict the average | 6,817 | — | — | 5,160 | — |
| 1 | **My 2024 notebook, exactly as it was** | 7,413 | **+8.7%** | 2 | 5,201 | +0.8% |
| 2 | + scaler bug fixed | 7,357 | +7.9% | 2 | 5,185 | +0.5% |
| 3 | + "last fold" bug fixed | 7,345 | +7.7% | 2 | 5,227 | +1.3% |
| 4 | LightGBM (Tweedie), raw features | 6,421 | −5.8% | 5 | 4,634 | −10.2% |
| 5 | **+ contribution-history features (final)** | **6,403** | **−6.1%** | **5** | **4,625** | **−10.4%** |

<p align="center"><img src="reports/figures/ladder.png" width="860"></p>

**EN —** So my 2024 model was worse than the average, and fixing its two bugs barely moves it. What
actually made the difference was changing the model itself.

**TR —** Kısacası 2024 modelim ortalamanın gerisindeydi ve iki bug'ını düzeltmek bunu neredeyse hiç
değiştirmiyor. Asıl farkı yaratan, modelin kendisini değiştirmek oldu.

## 🐞 What I got wrong / Nerede hata yaptım

### 1. I never checked it against the average / Ortalamayla hiç kıyaslamadım

**EN —** With RMSE, predicting the average is the best you can do without looking at any features.
Any real model has to beat that, and I simply never checked. When I replay my notebook under honest
validation, it's 8.7% worse than the average and only wins in 2 of the 5 months. The replay in
[`src/original.py`](src/original.py) gives exactly the same predictions my notebook printed back
then, so this really is my model and not a weaker copy of it.

**TR —** RMSE'de hiçbir özelliğe bakmadan yapabileceğin en iyi şey ortalamayı tahmin etmek. Gerçek bir
modelin bunu geçmesi gerekiyor, ben de bunu hiç kontrol etmemişim. Notebook'umu dürüst bir doğrulamayla
yeniden çalıştırınca ortalamadan %8,7 kötü çıkıyor ve 5 ayın sadece 2'sinde öne geçiyor.
[`src/original.py`](src/original.py) içindeki tekrar, notebook'umun o zaman yazdırdığı tahminlerin
aynısını veriyor; yani bu gerçekten benim modelim, zayıflatılmış bir kopyası değil.

### 2. My cross-validation made me feel better than I should have / Çapraz doğrulama beni boşuna rahatlattı

<p align="center"><img src="reports/figures/cv_illusion.png" width="720"></p>

**EN —** Back then I ran a shuffled 10-fold CV, got an RMSE of 7,015 and took it as a good sign. On
those same folds the model even looks 6.9% better than the average. The problem is that shuffling
mixes all the months together, so the model gets to learn from the future. On top of that, 15,412
customers show up more than once, so the same person ends up on both sides of a split. Once I
validate month by month, the way the real task works, the picture flips to 8.7% worse.

**TR —** O zaman karıştırılmış bir 10-fold CV çalıştırmış, 7.015 RMSE almış ve bunu iyiye işaret
saymıştım. Aynı fold'larda model ortalamadan %6,9 iyi bile görünüyor. Sorun şu: karıştırınca bütün
aylar birbirine giriyor ve model gelecekten öğrenmiş oluyor. Üstüne 15.412 müşteri birden fazla kez
geçiyor, yani aynı kişi bölmenin iki tarafına da düşebiliyor. Gerçek görevdeki gibi ay ay doğrulayınca
tablo tersine dönüyor ve model %8,7 kötü çıkıyor.

### 3. I rescaled the test data on its own / Test verisini kendi başına ölçekledim

**EN —** I wrote `scaler.fit_transform(df_test)` where it should have been `scaler.transform`. That
means the December data was scaled with its own averages instead of the training ones, so the model's
coefficients were being applied to numbers on a slightly different scale. The `month` column even
became 0 everywhere, which to the model looks like July. There's a test in
[`tests/test_pipeline.py`](tests/test_pipeline.py) that shows this one line ruining an otherwise
perfect model.

**TR —** `scaler.transform` yazmam gereken yere `scaler.fit_transform(df_test)` yazmışım. Yani Aralık
verisi eğitimin değil kendi ortalamalarıyla ölçeklenmiş ve modelin katsayıları biraz farklı ölçekteki
sayılara uygulanmış. `month` sütunu bile her yerde 0 olmuş, bu da model için Temmuz demek.
[`tests/test_pipeline.py`](tests/test_pipeline.py) içinde bu tek satırın aslında kusursuz çalışan bir
modeli nasıl bozduğunu gösteren bir test var.

### 4. The model from the last fold made my predictions / Tahminleri son fold'daki model yaptı

**EN —** My CV loop overwrote `model` every round, and I never trained it again on the full data. So
my submission came from whatever model was fitted in the tenth fold, which had only seen 90% of the
rows. Both of these bugs were real, but fixing them changes the score by less than 1%. They weren't
the main problem; the model was.

**TR —** CV döngüm her turda `model`'in üzerine yazmış, ben de onu sonunda tüm veriyle bir daha
eğitmemişim. Yani submission'ım onuncu fold'da eğitilen, satırların sadece %90'ını görmüş modelden
gelmiş. İki bug da gerçekti, ama düzeltince skor %1'den az değişiyor. Asıl sorun onlar değil, modelin
kendisiydi.

### 5. A good hunch, used the wrong way / Doğru sezgi, yanlış kullanım

<p align="center"><img src="reports/figures/repeat_signal.png" width="720"></p>

**EN —** After predicting, I replaced the prediction for customers I'd seen before with their last
positive contribution. The instinct behind it was right. People who contributed last time contribute
again 81% of the time, compared to 7% for those who didn't. But swapping in a copied amount is a big
bet on exactly the rows that decide RMSE, and it goes either way: removing it helps in June to October
(+7.7% → +1.7%) and hurts a lot in November (+1.3% → +9.2%). When I gave the same idea to LightGBM as
features instead, it got better at telling who will contribute (AUC 0.878 → 0.884), but the RMSE
didn't improve (−6.1% → −5.9%). Since I'd decided to pick the final model by RMSE, I left it out
(see [`reports/checks.csv`](reports/checks.csv)).

**TR —** Tahminden sonra, daha önce gördüğüm müşterilerin tahminini son pozitif katkılarıyla
değiştirmişim. Arkasındaki sezgi doğruydu: önceki sefer katkı yapanların %81'i yine yapıyor, yapmayanlarda
bu oran %7. Ama kopyalanmış bir tutarı olduğu gibi koymak, tam da RMSE'yi belirleyen satırlar üzerine
büyük bir bahis ve iki yöne de gidebiliyor. Kaldırınca Haziran–Ekim'de iyileşiyor (+%7,7 → +%1,7),
Kasım'da ise ciddi kötüleşiyor (+%1,3 → +%9,2). Aynı fikri LightGBM'e özellik olarak verince kimin
katkı yapacağını daha iyi ayırt ediyor (AUC 0,878 → 0,884), ama RMSE iyileşmiyor (−%6,1 → −%5,9). Final
modeli RMSE'ye göre seçeceğime baştan karar verdiğim için onu dışarıda bıraktım
([`reports/checks.csv`](reports/checks.csv)).

### 6. I trusted a single month / Tek bir aya güvendim

**EN —** In 2024 I tried around 1,000 feature combinations, training on March to October and scoring
only on November. The clearest thing it told me was to drop `RTRNDESVAMNT`. When I check that across
five months instead of one, the gain nearly disappears (+7.7% → +7.6%). One month on this data just
isn't enough to decide anything.

**TR —** 2024'te yaklaşık 1.000 özellik kombinasyonu denemiş, Mart–Ekim ile eğitip sadece Kasım'a göre
puanlamıştım. Bana en net söylediği şey `RTRNDESVAMNT`'yi çıkarmamdı. Bunu tek ay yerine beş ayda
kontrol edince kazanç neredeyse kayboluyor (+%7,7 → +%7,6). Bu veride tek bir aya bakıp karar vermek
yetmiyor.

## 🐋 Why the improvements look small / İyileşmeler neden küçük görünüyor

<p align="center"><img src="reports/figures/whale_share.png" width="720"></p>

**EN —** Nine out of ten rows are zero, and the few that aren't can get huge: the largest single
contribution is 1.26 million. In most months the 10 biggest rows (out of 15 to 34 thousand) make up
most of the squared error, and in April one row alone is about three quarters of it. Nobody can
predict a seven-figure top-up from these columns, so RMSE here mostly comes down to how badly you miss
a handful of big savers. That's why even a clearly better model only beats the average by 6 to 10%,
why most of the public leaderboard sat between 8,500 and 9,400, and why I stopped trusting any
single month.

**TR —** On satırdan dokuzu sıfır, sıfır olmayanlar ise çok büyüyebiliyor: tek seferde yatırılan en
büyük ek katkı 1,26 milyon. Çoğu ayda (15 ile 34 bin satır arasından) en büyük 10 satır karesel hatanın
büyük kısmını oluşturuyor, Nisan'da ise tek bir satır hatanın yaklaşık dörtte üçü. Bu sütunlardan yedi
haneli bir ek ödemeyi kimse tahmin edemez; o yüzden burada RMSE büyük ölçüde birkaç büyük tasarrufçuyu
ne kadar kaçırdığına bağlı. Belirgin şekilde daha iyi bir modelin bile ortalamayı ancak %6–10 geçmesi,
public leaderboard'un çoğunun 8.500 ile 9.400 arasında toplanması ve benim artık tek bir aya
güvenmemem bundan.

## 🔧 What I changed / Neleri değiştirdim

**EN —** I swapped linear regression for LightGBM with a Tweedie objective, which is made for data
that's mostly zeros with a long tail. It can't predict negative amounts, and one extreme value in a
column can't send a prediction flying the way it did in the linear model. I also stopped filling
missing values with 0. Seventeen of the 36 columns never go below exactly 100 (my guess is the
anonymisation added a constant), so a 0 was actually below the real floor; now LightGBM handles the
gaps itself. On top of that I added a few summaries of the 11 monthly contribution columns (average,
spread, maximum, how many months were paid, trend, and how they compare to the planned monthly
amount), and dropped `month` as a feature, since December never appears in training. Every one of
these choices, including the Tweedie setting of 1.8, was made on June to October only.

**TR —** Doğrusal regresyonu, çoğu sıfır ve uzun kuyruklu veriler için tasarlanmış Tweedie
objective'li LightGBM ile değiştirdim. Negatif tutar tahmin edemiyor ve bir sütundaki tek bir uç değer,
doğrusal modeldeki gibi tahmini uçuramıyor. Eksik değerleri 0 ile doldurmayı da bıraktım. 36 sütunun
17'si hiçbir zaman tam 100'ün altına inmiyor (tahminim, anonimleştirme sırasında sabit bir sayı
eklenmiş), yani 0 aslında gerçek tabanın altında kalıyordu; artık boşlukları LightGBM kendisi
yönetiyor. Bunlara ek olarak 11 aylık katkı sütununun birkaç özetini ekledim (ortalama, dağılım,
maksimum, kaç ay ödendiği, eğilim ve planlanan aylık tutarla karşılaştırma) ve Aralık eğitimde hiç
olmadığı için `month`'u özelliklerden çıkardım. Tweedie parametresi olan 1,8 dahil bu kararların
hepsini sadece Haziran–Ekim'e bakarak verdim.

<p align="center"><img src="reports/figures/feature_importance.png" width="560"></p>

## ⚠️ What I can't tell you yet / Henüz söyleyemediklerim

**EN —** The real December answers were never released, so I don't know where the new model would
land on the leaderboard. `python train.py` writes `submissions/submission_final.csv` (plus an exact
replay of my 2024 submission) so I can try a late submission. One thing I did notice: my 2024
submission predicted an average of **1,265** per customer for December, about 62% above the usual
783. The new model predicts **788**. With RMSE, being off on the average costs you on every single
row.

**TR —** Aralık'ın gerçek cevapları hiç yayımlanmadı, dolayısıyla yeni modelin leaderboard'da nereye
düşeceğini bilmiyorum. `python train.py`, geç gönderim deneyebilmem için
`submissions/submission_final.csv` dosyasını (yanına da 2024 submission'ımın birebir tekrarını)
yazıyor. Yine de fark ettiğim bir şey var: 2024 submission'ım Aralık için müşteri başına ortalama
**1.265** tahmin etmiş, bu da olağan 783'ün yaklaşık %62 üstü. Yeni model **788** diyor. RMSE'de
ortalamayı kaçırmak her bir satırda sana ceza olarak geri dönüyor.

## 🗂️ Project structure / Proje yapısı

```
garanti-bes-forecast/
├── data/raw/                        # train.csv, test_input.csv (not in the repo, see below)
├── notebooks/
│   ├── 00_original_2024.ipynb       # my competition notebook as it was (outputs cleared)
│   └── 01_what_went_wrong.ipynb     # the whole story, step by step
├── src/
│   ├── config.py                    # paths, validation months, LightGBM settings
│   ├── data.py                      # loading and tidying both CSVs
│   ├── original.py                  # my 2024 notebook as a function, with a switch per bug
│   ├── features.py                  # contribution summaries + customer history (no peeking ahead)
│   ├── model.py                     # the average baseline + LightGBM (Tweedie)
│   ├── validation.py                # month-by-month validation and scoring
│   ├── ladder.py                    # every version from 2024 to final, plus side checks
│   └── figures.py                   # the charts in this README
├── tests/test_pipeline.py           # leakage and bug tests on small made-up data
├── reports/                         # results.csv, checks.csv, figures/
├── models/metrics.json
├── train.py                         # runs everything (~3 min on my laptop)
└── requirements.txt
```

## 🚀 Running it / Çalıştırmak için

**EN —** The data comes from a private Kaggle competition (`garanti-bbva-data-day-case-study`), so
I can't share it here. If you have access, drop `train.csv` and `test_input.csv` into `data/raw/`.
The tests don't need it.

**TR —** Veri özel bir Kaggle yarışmasından (`garanti-bbva-data-day-case-study`) geliyor, o yüzden
burada paylaşamıyorum. Erişimin varsa `train.csv` ve `test_input.csv` dosyalarını `data/raw/` içine
koyman yeterli. Testler veriye ihtiyaç duymuyor.

```bash
git clone https://github.com/derinteke/garanti-bes-forecast.git
cd garanti-bes-forecast

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                            # 8 tests, no data needed
python train.py                      # every version + final model + charts + submissions
jupyter notebook notebooks/01_what_went_wrong.ipynb
```

## 🔭 What I'd do next / Sırada ne var

**EN —** First I want to make a late submission, so the "I don't know yet" section gets a real
number. After that I'd split the problem in two: will this customer contribute at all, which is where
the history features clearly help, and if so, how much. The second part needs a loss that doesn't get
thrown around by a few seven-figure rows.

**TR —** Önce geç gönderim yapmak istiyorum, böylece "henüz bilmiyorum" kısmına gerçek bir sayı
yazabilirim. Sonra problemi ikiye bölerdim: müşteri katkı yapacak mı (geçmiş özelliklerinin açıkça işe
yaradığı kısım bu) ve yapacaksa ne kadar. İkinci kısım için birkaç yedi haneli satırın savurmadığı bir
kayıp fonksiyonu gerekiyor.

## 📚 Data / Veri

**EN —** Garanti BBVA Data Day case study (Kaggle, 2024): 173,589 anonymised month-end snapshots of
155,404 BES customers from March to November 2018, with banking and pension-plan columns. The test
set is 16,978 customers in December 2018.

**TR —** Garanti BBVA Data Day vaka çalışması (Kaggle, 2024): Mart–Kasım 2018 arasında 155.404 BES
müşterisine ait 173.589 anonimleştirilmiş ay sonu kaydı; bankacılık ve emeklilik planı sütunları
içeriyor. Test seti Aralık 2018'deki 16.978 müşteri.

## 📄 License

MIT.
