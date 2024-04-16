import os
from flask import Flask, render_template, request, redirect
import csv
import nltk
import pandas
from werkzeug.utils import secure_filename
from sentimen import lower, remove_punctuation, remove_stopwords, stem_text, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from yellowbrick.text import TSNEVisualizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.stem import PorterStemmer
import pickle
import math
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
if __name__ == "__main__":
    app.run(debug=True)


# dashboard get data
@app.route('/', methods=['GET', 'POST'])
def index():
    if os.path.exists('uploads/dataset_clear.csv'):
        text = pandas.read_csv('uploads/dataset_clear.csv', encoding='latin-1')
        text.dropna(axis=0)
        positif, negatif= text['Labels'].value_counts()
        total = positif + negatif
        return render_template('index.html',total=total, positif=positif, negatif=negatif)
    else:
        return render_template('index.html')
    

#upload data
ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER']='uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/uploaddata', methods=['GET', 'POST'])
def uploaddata():
    if request.method == 'GET':
        if os.path.exists('uploads/dataset.csv'):
            text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1').head(100)
            # Inisialisasi list untuk menyimpan baris tabel HTML
            table_rows = []
                # Konversi setiap baris DataFrame menjadi baris HTML dan tambahkan ke list table_rows
            for index, row in text.iterrows():
                table_row = "<tr>"
                for value in row:
                    table_row += "<td>{}</td>".format(value)
                table_row += "</tr>"
                table_rows.append(table_row)
                # Render template dengan data yang disiapkan
            return render_template('uploaddata.html', table_rows=table_rows)
        else:
            return render_template('uploaddata.html')
    
    elif request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            file.filename = "dataset.csv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            # Reload the data after saving the file
            text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
            
            return render_template('uploaddata.html', tables=[text.to_html()])
            
@app.route('/delete/<int:index>', methods=['GET'])
def delete_data(index):
    if os.path.exists('uploads/dataset.csv'):
        text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
        text.drop(index=index-1, inplace=True)  # Hapus baris sesuai dengan indeks yang dipilih
        text.to_csv('uploads/dataset.csv', index=False)  # Simpan kembali file CSV tanpa baris yang dihapus
    return redirect('/uploaddata')


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if os.path.exists('uploads/dataset_stemmed.csv'):
        # Baca data CSV dan ambil 100 baris pertama
        text = pandas.read_csv('uploads/dataset_stemmed.csv', encoding='latin-1').head(100)
        # Inisialisasi list untuk menyimpan baris tabel HTML
        table_rows = []
        # Konversi setiap baris DataFrame menjadi baris HTML dan tambahkan ke list table_rows
        for index, row in text.iterrows():
            table_row = "<tr>"
            for value in row:
                table_row += "<td>{}</td>".format(value)
            table_row += "</tr>"
            table_rows.append(table_row)
        # Render template dengan data yang disiapkan
        return render_template('preprocessing.html', table_rows=table_rows)
    else:
        return render_template('preprocessing.html')

import pandas as pd

def normalize_text(text):
    # Load slang-formal mapping data
    slang_formal_data = pd.read_csv('slang_formal_mapping.csv')
    
    # Create a dictionary from slang to formal words
    slang_formal_dict = dict(zip(slang_formal_data['slang'], slang_formal_data['formal']))
    
    # Convert text to string if it's not already a string
    text = str(text)
    
    # Split text into words
    words = text.split()
    
    # Normalize each word using the slang-formal dictionary
    normalized_words = [slang_formal_dict.get(word, word) for word in words]
    
    # Join normalized words back into text
    normalized_text = ' '.join(normalized_words)
    
    return normalized_text


@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    # Membaca data dari file CSV
    text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
    
    # Lakukan preprocessing pada teks
    text['Text'] = text['Text'].apply(lambda x: preprocess_data(x))
    
    # Menyimpan hasil preprocessing ke dalam file CSV
    text.to_csv('uploads/dataset_clear.csv', index=False)

    # Membaca data yang sudah di-preprocessing dari file CSV
    text = pandas.read_csv('uploads/dataset_clear.csv', encoding='latin-1')

    # Normalisasi teks
    text['Normalisasi'] = text['Text'].apply(normalize_text)

    # Menyimpan hasil preprocessing ke dalam file CSV
    text.to_csv('uploads/normalisasi.csv', index=False)

    # Membaca data yang sudah di-normalisasi dari file CSV
    text = pandas.read_csv('uploads/normalisasi.csv', encoding='latin-1')

    # Tokenisasi teks dan tambahkan hasilnya ke dalam kolom baru 'Tokenized_Text'
    text['Tokenized_Text'] = text['Normalisasi'].apply(lambda x: word_tokenize(str(x)))

    # Stemming teks menggunakan PorterStemmer
    stemmer = PorterStemmer()
    text['Stemmed_Text'] = text['Tokenized_Text'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

    # Menyimpan hasil stemming ke dalam file Excel
    text.to_csv('uploads/dataset_stemmed.csv', index=False)
    text = pandas.read_csv('uploads/dataset_stemmed.csv', encoding='latin-1').head(100)
        # Inisialisasi list untuk menyimpan baris tabel HTML
    table_rows = []
        # Konversi setiap baris DataFrame menjadi baris HTML dan tambahkan ke list table_rows
    for index, row in text.iterrows():
        table_row = "<tr>"
        for value in row:
            table_row += "<td>{}</td>".format(value)
        table_row += "</tr>"
        table_rows.append(table_row)
        # Render template dengan data yang disiapkan
    return render_template('preprocessing.html', table_rows=table_rows)

@app.route('/tfidfpage', methods=['GET', 'POST'])
def tfidfpage():
    text_df = pandas.read_csv('uploads/normalisasi.csv', encoding='latin-1')

    # Menghapus baris dengan nilai NaN
    text_df.dropna(axis=0, inplace=True)

    # Mengambil kolom 'Normalisasi' sebagai list teks
    texts = text_df['Normalisasi'].tolist()

    # Menghitung TF-IDF untuk dokumen yang tersedia
    tfidf_dict = calculate_tfidf(texts)

    # Kirim hasil TF-IDF ke template HTML
    return render_template('tfidf.html', tfidf_dict=tfidf_dict, total=len(texts))

# @app.route('/normalisasi', methods=['GET', 'POST'])
# def normalisasi():
#         if os.path.exists('uploads/normalisasi.csv'):
#             text = pandas.read_csv('uploads/normalisasi.csv', encoding='latin-1').head(10)
#             return render_template('normalisasi.html', tables=[text.to_html()])
#         else:
#             return render_template('normalisasi.html')

# @app.route('/normalisasing', methods=['GET', 'POST'])
# def normalisasing():
#     text = pandas.read_csv('uploads/dataset_clear.csv', encoding='latin-1')
#     text['Text'] = text['Text'].apply(lambda x: normalize_text(x))
#     text.to_csv('uploads/normalisasi.csv', index=False, header=True)
#     return render_template('normalisasi.html', tables=[text.to_html(classes='table table-bordered', table_id='dataTable')])


def calculate_tfidf(texts):
    # Menghitung Term Frequency (TF) untuk setiap term dalam setiap dokumen
    tf_dict = {}
    doc_terms = {}  # Mengumpulkan semua term untuk setiap dokumen
    for idx, text in enumerate(texts):
        terms = text.split()
        term_count = len(terms)
        doc_terms[idx] = terms  # Simpan terms untuk dokumen ini
        for term in terms:
            if term not in tf_dict: 
                tf_dict[term] = {}
            if idx not in tf_dict[term]:
                tf_dict[term][idx] = 0  #idx = index
            tf_dict[term][idx] += 1 / term_count  # Menghitung TF

    # Menghitung Inverse Document Frequency (IDF) untuk setiap term
    doc_count = len(texts)
    idf_dict = {}
    for term in tf_dict:
        doc_freq = len(tf_dict[term])
        idf_dict[term] = math.log(doc_count / (doc_freq + 1))  # Menghitung IDF

    # Menghitung TF-IDF untuk setiap term dalam setiap dokumen
    tfidf_dict = {}
    for term in tf_dict:
        tfidf_dict[term] = {}
        for doc_idx in tf_dict[term]:
            doc_terms_str = ', '.join(doc_terms[doc_idx])  # Gabungkan terms ke dalam satu string
            if doc_terms_str not in tfidf_dict[term]:
                tfidf_dict[term][doc_terms_str] = 0
            tfidf_dict[term][doc_terms_str] += tf_dict[term][doc_idx] * idf_dict[term]

    return tfidf_dict

# TF_IDF BAWAAN
def data(text):
    text['Labels'] = text['Labels'].map({'positif': 1, 'negatif': 0})
    X = text['Normalisasi']   
    y = text['Labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@app.route('/tfidf', methods=['GET', 'POST'])
def tfidf():
    text = pandas.read_csv('uploads/normalisasi.csv', encoding='latin-1')
    text.dropna(axis=0, inplace=True)  # Membersihkan nilai np.nan

    positif, negatif= text['Labels'].value_counts()
    total = positif + negatif

    X_train, X_test, y_train, y_test = data(text)

    # Inisialisasi vektorisator TF-IDF
    vectorizer = TfidfVectorizer()

    # Lakukan vektorisasi TF-IDF pada data teks yang telah dibersihkan
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
  
    # Menyimpan vektorisator ke disk
    pickle.dump(vectorizer, open('uploads/vectorizer.model','wb'))

    text_df = pandas.read_csv('uploads/normalisasi.csv', encoding='latin-1')

    # Menghapus baris dengan nilai NaN
    text_df.dropna(axis=0, inplace=True)

    # Mengambil kolom 'Normalisasi' sebagai list teks
    texts = text_df['Normalisasi'].tolist()

    # Menghitung TF-IDF untuk dokumen yang tersedia
    tfidf_dict = calculate_tfidf(texts)
    # Kirim hasil TF-IDF ke template HTML
    return render_template('tfidf.html', tfidf_dict=tfidf_dict, total=len(texts))



@app.route('/klasifikasisvm1', methods=['GET', 'POST'])
def klasifikasisvm1():

    return render_template ('klasifikasisvm.html')


@app.route('/klasifikasisvm', methods=['GET', 'POST'])
def klasifikasisvm():
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
    
    # Loading model to compare the results
    vectorizer = pickle.load(open('uploads/vectorizer.model','rb'))

    text = pandas.read_csv('uploads/normalisasi.csv', encoding='latin-1')
    text.dropna(axis=0, inplace=True)  # Membersihkan nilai np.nan

    X_train, X_test, y_train, y_test = data(text)

    # Lakukan vektorisasi TF-IDF pada data teks yang telah dibersihkan
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Process of making models Klasifikasi SVM LINEAR
    linear = SVC(kernel="linear")
    linear.fit(X_train, y_train)
    linear_pred = linear.predict(X_test)

    # Process of making models Klasifikasi SVM RBF
    rbf = SVC(class_weight=None, C=1, gamma=0.001, kernel='rbf', random_state=100)
    rbf.fit(X_train, y_train)
    rbf_pred = rbf.predict(X_test)

    # Saving models to disk
    pickle.dump(linear, open('uploads/linear.model','wb'))
    pickle.dump(rbf, open('uploads/rbf.model','wb'))

    # Calculating evaluation metrics for linear kernel
    f1_score_linear = f1_score(y_test, linear_pred)
    accuracy_score_linear = accuracy_score(y_test, linear_pred) * 100  # Convert to percentage
    precision_score_linear = precision_score(y_test, linear_pred) * 100  # Convert to percentage
    recall_score_linear = recall_score(y_test, linear_pred) * 100  # Convert to percentage
    tn_linear, fp_linear, fn_linear, tp_linear = confusion_matrix(y_test, linear_pred).ravel()

    # Calculating evaluation metrics for rbf kernel
    f1_score_rbf = f1_score(y_test, rbf_pred)
    accuracy_score_rbf = accuracy_score(y_test, rbf_pred) * 100  # Convert to percentage
    precision_score_rbf = precision_score(y_test, rbf_pred) * 100  # Convert to percentage
    recall_score_rbf = recall_score(y_test, rbf_pred) * 100  # Convert to percentage
    tn_rbf, fp_rbf, fn_rbf, tp_rbf = confusion_matrix(y_test, rbf_pred).ravel()

    return render_template ('klasifikasisvm.html', f1_score_linear=f1_score_linear, accuracy_score_linear=accuracy_score_linear, precision_score_linear=precision_score_linear, recall_score_linear=recall_score_linear, 
    tn_linear=tn_linear, fp_linear=fp_linear, fn_linear=fn_linear, tp_linear=tp_linear, f1_score_rbf=f1_score_rbf, accuracy_score_rbf=accuracy_score_rbf, precision_score_rbf=precision_score_rbf, 
    recall_score_rbf=recall_score_rbf, tn_rbf=tn_rbf, fp_rbf=fp_rbf, fn_rbf=fn_rbf, tp_rbf=tp_rbf)




@app.route('/tesmodel1', methods=['GET', 'POST'])
def tesmodel1():
    results = []
    with open('results.txt', 'r') as file:
        for line in file:
            original_text, preprocessed_text, sentiment = line.strip().split('\t')
            results.append({'original_text': original_text, 'preprocessed_text': preprocessed_text, 'sentiment': sentiment})

        # Reverse the results list
    results.reverse()
    return render_template ('tesmodel.html', results=results)


@app.route('/tesmodel', methods=['GET', 'POST'])
def tesmodel():
    # Loading model to compare the results
    model = pickle.load(open('uploads/rbf.model','rb'))
    vectorizer = pickle.load(open('uploads/vectorizer.model','rb'))

    text = request.form['text']
    original_text = text

    hasilprepro = preprocess_data(text)
    hasiltfidf = vectorizer.transform([hasilprepro])

    # cek prediksi dari kalimat
    hasilsvm = model.predict(hasiltfidf)
    if hasilsvm == 0:
        hasilsvm = 'NEGATIF'
    else:
        hasilsvm = 'POSITIF'
    
    # Save results to a text file in tabular format
    with open('results.txt', 'a') as file:
        file.write(f"{original_text}\t{hasilprepro}\t{hasilsvm}\n")

    # Read the contents of the results.txt file and pass them to the template
    results = []
    with open('results.txt', 'r') as file:
        for line in file:
            original_text, preprocessed_text, sentiment = line.strip().split('\t')
            results.append({'original_text': original_text, 'preprocessed_text': preprocessed_text, 'sentiment': sentiment})

    # Reverse the results list
    results.reverse()

    return render_template('tesmodel.html', results=results)


