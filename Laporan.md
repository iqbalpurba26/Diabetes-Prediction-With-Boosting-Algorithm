# Laporan Proyek Machine Learning - M. Iqbal Purba

## Domain Proyek

Diabetes Mellitus merupakan penyakit gangguan metabolik akibat Pankreas (organ di elakang perut) memproduksi sedikit insulin atau tidak ada insulin sama sekali [[1]](https://jurnal.untan.ac.id/index.php/jepin/article/view/40718/75676587680). Fakta mengarah pada betapa diabetes telah menjadi salah satu penyebab kematian terbesar di dunia [[2]](https://books.google.co.id/books?hl=id&lr=&id=KxdIDwAAQBAJ&oi=fnd&pg=PP1&dq=apa+itu+diabetes&ots=oeBk7j3kPr&sig=O3o3Vgb1RW5iLR6QXkl7mJalVQY&redir_esc=y#v=onepage&q=apa%20itu%20diabetes&f=false). Oleh karena itu, peningkatan efektivitas dalam deteksi dini dan prediksi penyakit diabetes sangat penting untuk mengurangi beban kesehatan dan meningkatkan manajemen penyakit secara keseluruhan.

Ada banyak jenis diabetes; yaitu paling umum adalah diabetes tipe 1 dan tipe 2 [[1]](https://jurnal.untan.ac.id/index.php/jepin/article/view/40718/75676587680)

1. Diabetes Tipe 1 : Ini hasil dari tidak membuat insulin. Penderita diabetes tipe 1 membutuhkan insulin, baik dengan injeksi atau menggunakan pompa insulin.
2. Diabetes Tipe 2 : Ini hasil dari resistensi insulin, dimana sel-sel gagal menggunakan insulin dengan benar.

Pemeriksaan penyakit diabetes pada kedokteran biasanya dilakukan dengan cara diagnosis penyakit menggunakan hasil uji laboratorium dan rekam medis gejala penyakit. Guna menekan angka kematian angka kematian dari penyakit diabetes ini, para pakar kesehatan harus melakukan diagnosis penyakit sedini mungkin [[3]](https://drive.google.com/file/d/1OcJ2eV465DAivRIe2jilQOnyw6_LxQJl/view?usp=sharing). Hal ini serupa dengan penelitian yang dilakukan oleh [[4]](https://jurnal.polsri.ac.id/index.php/teknika/article/view/6808) yang menyatakan bahwa upaya lain yang dapat dilakukan untuk membantu mengatasi penyakit diabetes adalah mengembangkan sebuah model melalui penerapan teknologi komputer yang mampu melakukan prediksi penyakit diabetes, sebagai upaya mendeteksi penyakit diabetes sejak dini.

Penelitian terdahulu yang telah berhasil dikembangkan dengan menerapkan algoritma _machine learning_ diantaranya menggunakan metode _Grid Search_ pada algoritma _Logistic Regression_ dengan tingkat akurasi sebesar 83.33% [[2]](https://jurnal.untan.ac.id/index.php/jepin/article/view/40718/75676587680). Selain itu terdapat juga penelitian yang dilakukan untuk mengklasifikasikan penyakit jantung menggunakan algoritma _Logistic Regression_ didapatkan tingkat akurasi sebesar 83% [[5]](https://jurnal.untan.ac.id/index.php/jepin/article/view/48053/75676591338). Pada penelitian lain digunakan algoritma _Naive Bayes_ untuk mengklasifikasikan penyakit diabetes, mendapatkan tingkat akurasi sebesar 90,20% [[6]](https://jurnal.tau.ac.id/index.php/siskom-kb/article/view/169/146).

## Business Understanding

### Problem Statements

- Bagaimana pengaruh dari setiap features terhadap penyakit diabetes?
- Feature apa yang paling mempengaruhi penyakit diabetes?
- Bagaimana perbandingan akurasi dari algoritma _Support Vector Classifier, Random Forest Classifier, Logistic Regression_, dan _AdaBoost Classifier_?

### Goals

Tujuan :

- Untuk mengetahui bagaimana setiap feature mempengaruhi hasil diabetes.
- Untuk mengetahui feature paling berpengaruh terhadap penyakit diabetes.
- Untuk mengetahui perbandingan dari algoritma _Support Vector Classifier, Random Forest Classifier, Logistic Regression_, dan _AdaBoost Classifier_ sehingga dapat ditentukan algoritma paling efektif pada dataset tersebut.

### Solution statements

Untuk mendapatkan tujuan 1 dan 2 yang telah dijelaskan diatas, maka dilakukan Exploratory Data Analysis (EDA) pada dataset tersebut. Setiap fitur kategorikal, akan divisualisasikan terhadap kolom target yaitu kolom 'diabetes'. Kemudian dilakukan analisa terhadap hasil visualisasi. Begitu juga dengan kolom numerical, dilakukan visualisasi data dengan pairplot dan melihat korelasi antara kolom dengan kolom target menggunakan heatmap.

Sedangkan untuk tujuan ketiga, dilakukan dengan melatih model menggunakan 4 algoritma yaitu _Support Vector Classifier, Random Forest Classifier, Logistic Regression_, dan _AdaBoost Classifier_. Keempat algoritma tersebut dilatih menggunakan dataset yang serupa dan dengan perbandingan data latih dan data uji yang sama. Setelah dilakukan pelatihan model, maka selanjutnya adalah mengevaluasi setiap model yang telah dibentuk menggunakan metriks evaluasi _accuracy_score_. Dengan metriks evaluasi _accuracy_score_ tersebut akan dipilih 1 algoritma dengan nilai _accuracy_score_ tertinggi. Kemudian dilakukan testing menggunakan salah satu data yang ada di dataset untuk memastikan keakuratan model.

## Data Understanding

Data yang digunakan dalam projek ini adalah dataset yang terdiri dari 100,000 data dengan 9 kolom. Terdapat 8 kolom independen dan 1 kolom dependen (target). Kolom independen diantaranya adalah _'gender'_, _'age'_, _'hypertension'_, _'heart_disease'_, _'smoking_history'_, _'bmi'_, _'HbA1c_level'_, dan _'blood_glucose_level'_. Sedangkn kolom dependen/target pada dataset tersebut adalah 'diabetes'. Dataset tersebut diambil melalui _public dataset_ Kaggle yang dapat diunduh pada link berikut : [Diabetes Prediction](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data).

### Variabel-variabel pada _Diabetes Prediction dataset_ adalah sebagai berikut:

- _gender_ : Menampilkan jenis kelamin dari sampel. Kolom _'gender'_ memiliki 3 nilai yaitu _Female_, _Male_ dan _Other_.
- _age_ : Menampilkan umur dari setiap sampel. Umur merupakan salah satu faktor penting dalam prediksi penyakit diabetes. Pada kolom ini, rentang umur sampel antara 0-80 tahun.
- _hypertension_ : Menampilkan riwayat penyakit hipertensi dari setiap sampel. Jika sampel memiliki hipertensi maka ditandai dengan angka 1, dan jika tidak memiliki riwayat hipertensi ditandai dengan angka 0.
- _heart_disease_ : Menampilkan riwayat penyakit jantung dari setiap sampel. Jika sampel memiliki riwayat penyakit jantung ditandai dnegan angka 1, dan jika tidak memiliki riwayat penyakit jantung ditandai dengan angka 0.
- _smoking_history_ : Menampilkan riwayat merokok sampel. Pada kolom ini terdapat 6 nilai bertipe object, yaitu _No Info, current, ever, former, never_ dan _not current_.
- _bmi_ : Menampilkan nilai _BMI_ dari setiap sampel. _BMI_ merupakan singkatan dari _Body Mass Index_ dimana ini adalah ukuran lemak tubuh berdasarkan berat dan tinggi badan.
- _HbA1c_level_ : Menampilkan kadar _HbA1C_ atau Hemoglobin A1c pada tubuh seseorang selama 2-3 bulan terakhir. Dengan kata lain, kolom ini berisi ukuran rata-rata kadar gula darah seseorang selama 2-3 bulan terakhir.
- _blood_glucose_level_ : Menampilkan kadar glukosa darah yang mengacu pada jumlah glukosa dalam aliran darah pada waktu tertentu.
- _diabetes_ : Menampilkan apakah seseorang tersebut terkena diabetes atau tidak. Jika terkena, maka ditandai dengan angka 1, dan jika tidak maka ditandai dengan angka 0.

Untuk memahami data tersebut, maka digunakan teknik _Exploratory Data Analysis_ (EDA) dan visualisasi data. Sebelum melakukan visualisasi, maka dilakukan pengecekan _missing value_, _duplicated data_, dan _outliers_. Pada pengecekan, tidak ditemukannya _missing value_, namun ditemukan sebanyak 3854 data yang duplikat. Sedangkan untuk outliers, ditemukan beberapa data yang dikategorikan sebagai outliers menggunakan boxplot. Kemudian dilakukan pembersihan data terhadap data yang duplikat dan _outliers_. Untuk menangani duplikat data, maka dilakukan penghapusan menggunakan ` df.drop_duplicates(inplace=True)`. Data yang tersisa setelah dihapus duplikat adalah sebanyak 96,146 data. Selanjutnya dilakukan penanganan _outliers_ dan menyisakan data sebanyak 88,195 data. Jumlah daa 88,195 tersebut merupakan data yang akan diproses dan diolah untuk melatih model.

Proses _Exploratory Data Analysis_ pun dilakukan. Pada proses ini, dilakukan analisis hubungan kolom independen yang bersifat kategorikal dengan kolom dependen/target yaitu kolom 'diabetes'. Hasil EDA kolom kategorikal dapat dilihat pada gambar 1.

![eda1](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/a08ee301-4ebe-4fbd-bde6-eec336b5d1a7) Gambar 1 : Hasil Exploratory Data Analysis Kolom Kategorikal

Dari gambar 1 diatas, dapat dilihat bahwa yang paling mempengaruhi adalah riwayat hipertensi ('hypertension') pasien. hampir 80000 pasien yang memiliki hipertensi, positif diabetes.

Kolom lainnya yang memiliki keterikatan kuat terhadap hasil diabetes adalah riwayat merokok, walaupun pada grafik terlihat bahwa orang yang tidak pernah merokok ('never') lebih banyak positif. terbukti bahwa mantan perokok ('former') mengidap penyakit diabetes itu hampir 30000. Hal ini menunjukkan bahwa 2/3 dari pengidap penyakit diabetes, pernah merokok.

Sedangkan lebih dari 1/2 pengidap diabetes berjenis kelamin perempuan. Riwayat penyakit jantung tidak memiliki pengaruh yang signifikan terhadap penyakit diabetes. Hal ini terbukti bahwa lebih dari 80000 sampel tidak memiliki riwayat hipertensi terkena penyakit diabetes.

Kemudian untuk mendapatkan insight yang lebih mendalam, dilakukan kategorisasi terhadap kolom BMI yang sebenarnya bertipe numerik. Namun dibuat kategorinya dengan ketentuan :

- BMI <= 18.5 : _Underweight_
- 18.5 < BMI <= 24.9 : _Normal_
- 24.9 < BMI <= 29.9 : _Overweight_
- BMI > 29.9 : _Obesity_

Hasil visualisasi dapat dilihat pada gambar 2.

![eda2](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/7176776d-1529-4fd9-ad6b-eca7e052f1f7) Gambar 2 : Kategori BMI terhadap diabetes

Pada gambar 2, dikategorikan nilai BMI kedalam 4 kategori yaitu _'underweight', 'normal', 'overweight'_ dan _'obesity'_. Gambar menunjukkan bahwa 47% dari total sampel positif diabetes dengan tipe bmi 'overweight'. Hal ini berarti BMI berpengaruh besar terhadap hasil diabetes.

Tidak hanya BMI, kolom _'blood_glucose_level'_ yang sebenarnya kolom numerik, di-_improve_ menjadi kategorical dengan ketentuan :

- _'blood_glucose_level'_ <= 99 : _Normal_
- 99 < _'blood_glucose_level'_ <= 125 : _Prediabetes_
- _'blood_glucose_level'_ > 125 : _Diabetes_

Hasil visualisasi dapat dilihat pada gambar 3.

![eda3](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/3b632624-13dc-4129-a1a3-42d8716f9769) Gambar 3 : Kategori Level Gula Darah terhadap diabetes.

Dari gambar 3 dapat dilihat bahwa sampel memiliki kemungkinan positif diabetes jika level gula darahnya tergolong _Diabetes_. Sedangkan level gula darah _Normal_ dan _Prediabetes_ 100% tidak terkena diabetes. Hal ini membuktikan bahwa nilai _'blood_glucose_level'_ berpengaruh terhadap prediksi diabetes.

Sedangkan kolom yang bertipe numerikal, akan divisualisasikan menggunakan 3 jenis plot yaitu _hist, pairplot_, dan _heatmap_.

Untuk gambar plot _hist_ dapat dilihat pada gambar 4.

![eda4](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/d9c4f0d0-cac1-42a9-80ce-26081fc1c4f5) Gambar 4 : _Hist plot_ kolom numerikal

Pada gambar 4 diatas, dapat disimpulkan beberapa hal yaitu :

- Pada rentang usia 77 - 80 tahun, memiliki jumlah positif diabetes terbesar.
- BMI pada rentang 25-30, positif diabetes hampir 30000 sampel
- level HbA1c_level pada rentang 0-6.5, memiliki sampel positif lebih dari 1/2 jumlah sampel
- level blood_glucose_level pada rentang 150-160, memiliki sampel positif diabetes lebih banyak dibandingakan yang lainnya.

Untuk gambar _pair plot_ yang dilakukan, dapat dilihat pada gambar 5.

![eda5](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/cc0625c7-73fc-48c7-95e9-c692f21074fb) Gambar 5: _Pairplot_ kolom numerikal

Pada hasil gambar 5 diatas, dominan titik titik pada pairplot nya memiliki korelasi positif. Hal ini terbukti pada titik titiknya berjejer ke kanan sehingga disimpulkan bahwa dominan sesama kolom numerikal berkorelasi positif.

Sedangkan gambar _heatmap plot_ dapat dilihat pada gambar 6.

![eda6](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/f48d744f-ca71-4bc4-a7a0-0b4cfa170563) Gambar 6 : _Heatmap plot_ antar kolom numerikal

Pada gambar 6 diatas, kolom 'diabetes' memiliki korelasi paling tinggi dengan kolom _'HbA1c_level'_. Selanjutnya diikuti oleh kolom _'age'_ dan _'blood_glucose_level'_.

Sedangkan korelasi paling rendah dengan kolom 'diabetes' adalah kolom _'smoking_history_not_current'_ dan kolom _'smoking_history_current'_.

## Data Preparation

Pada preparation data, tidak terlalu banyak yang diterapkan. Pada saat proses _Exploratory Data Analysis_ dan visualisasi data dilakukan, telah diterapkan beberapa teknik preparation seperti pelabelan terhadap data yang bertipe object. Jadi data yang digunakan di EDA dicopy kedalam dataset baru yang digunakan untuk pembuatan model nantinya.

Data yang bertipe data object seperti _'gender'_ dan _smoking_history_ dilakukan pelabelan. Pada kolom _'gender'_, yang bernilai _'Female'_ diubah nilainya menjadi 0, _'Male'_ nilainya menjadi 1, dan _'Other'_ nilainya menjadi 2. Sedangkan pada kolom _smoking_history_ dilakukan teknik OneHotEncoder untuk melabeli secara otomatis dengan menambah kolom pada dataset.

Tujuan dari pelabelan tersebut yaitu:

1. memungkinkan model machine learning untuk memprediksi. Suatu model machine learning (kecuali nlp), hanya memahami nilai numerik.
2. Meningkatkan kerja model. Dengan menggunakan data numerik, maka kinerja model akan lebih efisien dalam memahami pola dalam data.
3. Interpretasi yang lebih baik. Dengan mengubah nilai string kedalam numerik, maka dapat lebih mudah mengidentifikasi kontribusi relatif dari setiap fitur terhadap hasil prediksi.

Kemudian dipisahkan kolom independen dan kolom dependen menggunakan 2 variabel baru yaitu X dan y. Variabel X akan menampung nilai kolom independen. Sedangkan variabel y akan menampung nilai kolom dependen/target yaitu kolom _'diabetes'_.

Setelah itu, proses preparation yang dilakukan selanjutnya adalah skalasi menggunakan standardscaler(). Pertama yang dilakukan adalah mengimport library StandardScaler(), kemudian memasukkan variabel X saja untuk dilakukan skalasi.Tujuan penggunakan library ini adalah untuk meningkatkan performa model. Beberapa model machine learning, sangat sensitif terhadap perbedaan skala fiturnya. Selain itu, jika dilakukan skalasi maka bobot yang dihasilkan oleh model memberikan gambaran yang relatif benar dan akurat terhadap hasil prediksi. Ada beberapa alasan memilih StandardScaler() digunakan untuk menskalasi data yaitu :

- Menghasilkan distribusi normal. StandardScaler() akan melakukan transformasi data sehingga variabel-variabelnya memiliki rata-rata 0 dan deviasi standar 1.
- Mengurangi pengaruh _outlier_. StandardScaler() menghitung deviasi standar yang tidak terpengaruh oleh nilai-nilai ekstrem.
- Memudahkan perbandingan koefisien. Banyak algoritma memberikan wawasan yang lebih baik ketika fitur-fiturnya telah di skalasi menggunakan StandardScaler. Koefisien yang dihasilkan akan mencerminkan seberapa besar dampak fitur terhadap prediksi, karena setiap fitur memiliki skala yang seragam.

Selanjutnya data yang telah diskalasi tersebut akan di transform agar berbentuk matriks. Kemudian matriks yang diskalasi tersebut disimpan kedalam variabel X, dan kolom target kedalam variabel y. Kemudian dilakukan pembagian data latih dan data testing. Perbandingan data latih dan data testing yang digunakan adalah 80%:20%.

## Modeling

Pada proses modelling, digunakan 4 algoritma yaitu Support Vector Classifier, Random Forest, Logistic Regression, dan AdaBoost Classifier. Adapun kelebihan dan kekurangan algoritma tersebut yaitu

- Support Vector Classifier

Kelebihan :

1. Mampu menangani outliers
2. Efektif dalam ruangan berdimensi tinggi (jumlah features lebih banyak daripada jumlah sampel)
3. Mampu menangani keputusan yang rumit.

Kekurangan :

1. Sensitif terhadap parameter dan kernel
2. Kinerja menurun pada dataset yang besar

- Random Forest

Kelebihan :

1. Tahan terhadap overfitting
2. Mampu menangai data yang tidak linear
3. Tidak perlu pemrosesan lanjutan

Kekurangan :

1. Komputasi memakan waktu
2. Kurang mudah diinterpretasikan

- Logistic Regression

Kelebihan :

1. Sederhana dan Cepat
2. Interpretabilitas Tinggi

Kekurangan :

1. Mampu menangani hubungan linear
2. Tidak mampu menangani fitur non linear secara alami

- AdaBoost Classifier

Kelebihan :

1. Mampu menangani model lemah
2. Tidak perlu menyesuaikan parameter yang rumit

Kekurangan :

1. Rentan terhadap noise dan outlier
2. Komputasi yang memakan waktu

Pada algoritma _Support Vector Classifier_, parameter yang digunakan adalah _kernel="linear"_. Berikut potongan _code_ nya
`svc_model = svm.SVC(kernel='linear')`. Cara kerja kernel linear ini akan memisahkan kelas dengan garis linear. kernel linear cocok untuk masalah klasifikasi dimana kelas-kelas dapat dipisahkan dengan baik oleh garis lurus dalam ruang fitur.

Pada algoritma Random Forest Classifier, digunakan parameter _n_estimators=50_ dan _random_state=2_. Hal ini dapat dilihat pada potongan _code_ berikut :
`rf_model = RandomForestClassifier(n_estimators=50, random_state=2)`. _n_estimators_ berfungsi untuk mengatur berapa jumlah pohon keputusan yang akan dijalankan, disini diatur nilainya menjadi 50. Sedangkan _random_state_ diatur untuk mereproduksi hasil yang konsisten. _Random Forest Classifier_ akan melibatkan pembentukan pohon keputusan secara acak, dan agregasi hasil prediksi dari setiap pohon tersebut untuk menghasilkan prediksi final.

Algoritma _Logistic Regression_ bekerja dengan menemukan parameter-parameter yang menggambarkan hubungan logistik antara variabel input (fitur) dan variabel output (kelas). Perhatikan potongan code berikut `lr_model = LogisticRegression(solver='liblinear', penalty='l1')`. Parameter yang diberikan yaitu _solver='libliner'_ dan regulasi L1 yang ditandai dengan `penalty='l1`.

Algoritma _AdaBoost Classifier_ memiliki parameter yang sama seperti algoritma _RandomForest Classifier_. Pada algoritma _AdaBoost Classifier_ tersebut, ditentukan parameter *n*estimators**nya 50 dan *random_state*nya 2.

Dari keempat algoritma tersebut, dipilih lah algoritma _Random Forest Classifier_ sebagai solusi dari _problem statement_ yang didefinisikan. Ada beberapa alasan pemilihan algoritma tersebut diantaranya yaitu :

- Memiliki performa tinggi. RFC memberikan performa yang tinggi dalam berbagai tugas klasifikasi dan regresi. Hal ini dapat menjadi solusi dengan ketepatan prediksi yang dilakukan.
- Stabilitas dan Redundansi. RFC cenderung lebih stabil dan kurang rentan terhadap _overfitting_ karena menggabungkan hasil dari banyak pohon keputusan yang dibangun secara acak.
- Tahan terhadap _oulier_ dan data _noisy_. Hasil prediksi dari RFC diambil berdasarkan mayoritas suara atau rata-rata dari banyak pohon keputusan. RFC lebih cenderung tahan terhadap _outlier_ dan data yang _noisy_
- Mudah digunakan. RFC mudah diimplementasikan dan digunakan terutama dengan menggunakan _library machine learning_ yang populer seperti _scikit-learn_ pada Python.

## Evaluation

Evaluasi yang digunakan adalah menggunakan metriks evaluasi _accuracy_score_. Pada projek tersebut, berikut dilampirkan gambar 7 perbandingan accuracy_score dari setiap algoritma yang digunakan

![accuracy_score](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/0a97fc79-f76c-4332-b16f-6eb6a7c9841c) Gambar 7 : Perbandingan _accuracy_score_ yang diperoleh

Pada gambar 7, dapat dilihat bahwa semua algoritma memiliki _accuracy_score_ diatas 95%. Sehingga dapat disimpulkan bahwa keempat model tersebut sudah dapat digunakan dalam implementasi nyatanya. Algoritma _Random Forest Classifier_ mendapatkan nilai _Train Accuracy_ tertinggi yaitu 99.8852%. Sedangkan _Train Accuracy_ terkecil diantara tertinggi adalah _Support Vector Classifier_ dengan nilai 96.1265%.

Untuk memahami metiks _accuracy_score_ yang digunakan, dapat dilihat pada gambar 8 berikut

![Alt text](https://wiki.cloudfactory.com/media/pages/docs/mp-wiki/metrics/accuracy/bc5dda9c32-1684142766/12.webp) Gambar 8 : Formula Metriks _Accuracy_

Keterangan :
True Positives (TP): Jumlah sampel positif yang diklasifikasikan dengan benar sebagai positif oleh model.

True Negatives (TN): Jumlah sampel negatif yang diklasifikasikan dengan benar sebagai negatif oleh model.

False Positives (FP): Jumlah sampel negatif yang salah diklasifikasikan sebagai positif oleh model.

False Negatives (FN): Jumlah sampel positif yang salah diklasifikasikan sebagai negatif oleh model.

Dengan kata lain, accuracy_score dihitung dengan membagi jumlah prediksi benar (True Positives + True Negatives) dengan total jumlah prediksi yang dilakukan oleh model.

Selanjutnya dilakukan pengujian langsung pada data yang ada di dataset. Digunakan data pada baris pertama untuk menguji keempat algoritma tersebut. Hasil pengujiannya dapat dilihat pada gambar 8 berikut.

![evaluasi](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/3834df9f-1955-4301-b3a4-58f5b0c95573) Gambar 8 : Hasil Evaluasi Pengujian Model

Dari gambar 8 tersebut, disimpulkan bahwa, keempat algoritma yang digunakan, memprediksi hasil yang sama dan benar. Yang membedakan keempat dari algoritma tersebut adalah hanya tingkat *accuracy_score*nya.

# REFERENSI

[1] M. I. Gunawan, D. Sugiarto and I. Mardianto, "Peningkatan Kinerja Akurasi Prediksi Penyakit Diabetes Mellitus Menggunakan Metode Grid Search pada Algoritma Logistic Regression," Jurnal Edukasi dan Penelitian Informatika, vol. 6, no. 3, pp. 280-284, 2020.

[2] H. Tandra, Diabetes Bisa Sembuh, Jakarta: PT Gramedia Pustaka Utama, 2015.

[3] A. Supandi, A. Faqih and F. M. Basysyar, "PREDIKSI PENYAKIT DIABETES MENGGUNAKAN MACHINE LEARNING DENGAN ALGORITMA NAIVE BAYES," Journal Sistem Informasi dan Manajemen, vol. 10, no. 2, pp. 146-152, 2022.

[4] Sriyanto and A. R. Supriyatna, "Prediksi Penyakit Diabetes Menggunakan Algoritma Random Forest," Jurnal Teknika, vol. 17, no. 1, pp. 163-172, 2023.

[5] F. Handayani, K. S. Kusuma, H. L. Asbudi, R. G. Purasiwi, R. Kusum, A. Sunyoto and W. M. Pradnya, "Komprasi Support Vector Machine, Logistic Regression, Dan Artificial Neural Network dalam Prediksi Penyakit Jantung," Jurnal Edukasi dan Penelitian Informatika, vol. 7, no. 3, pp. 329-334, 2021.

[6] A. Ridwan, "Penerapan Algoritma Naive Bayes Untuk Klasifikasi Penyakit Diabetes Melitus," Jurnal Sistem Komputer dan Kecerdasan Buatan, vol. 4, no. 1, pp. 15-21, 2020.
