import pandas as pd
import numpy as np
import os
import pickle
import re
import pathlib
from config import config
import psycopg2
import threading
import time
import string
from string import punctuation
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
import mysql.connector as mysql
import datetime
import save_models
import model_transactions as mt

base_dir = pathlib.Path(__file__).parent.absolute()
os.chdir(base_dir)
base_cwd = os.getcwd()
Accuracy_list = []
Model_list = []
threads = []

def text_processing(message):
    message = message.lower()
    Stopwords = stopwords.words('english')
    # Check characters to see if they are in punctuation
    no_punctuation = [char for char in message if char not in string.punctuation]

    # Join the characters again to form the string.
    no_punctuation = ''.join(no_punctuation)
    
    # Now just remove any stopwords
    return ' '.join([word for word in no_punctuation.split() if word.lower() not in Stopwords])

def change_cwd_data(customer):
    #cwd = os.getcwd()
    new_cwd = os.path.join(base_cwd, customer, "customer_data")
    os.chdir(new_cwd)

def change_cwd_model(customer,cur_dir):
    #cwd = os.getcwd()
    #dev_dir = save_models.get_dev_dir(customer)
    new_cwd = os.path.join(base_cwd, customer, cur_dir, "models")
    os.chdir(new_cwd)

def change_cwd_file(customer,cur_dir):
    #cwd = os.getcwd()
    #dev_dir = save_models.get_dev_dir(customer)
    new_cwd = os.path.join(base_cwd, customer, cur_dir, "files")
    os.chdir(new_cwd)
    
def change_cwd_model_prod(customer):
    #cwd = os.getcwd()
    prod_dir = save_models.get_prod_dir(customer)
    new_cwd = os.path.join(base_cwd, customer, prod_dir, "models")
    os.chdir(new_cwd)

def change_cwd_file_prod(customer):
    #cwd = os.getcwd()
    prod_dir = save_models.get_prod_dir(customer)
    new_cwd = os.path.join(base_cwd, customer, prod_dir, "files")
    os.chdir(new_cwd)

def change_cwd_model_dev(customer):
    #cwd = os.getcwd()
    dev_dir = save_models.get_dev_dir(customer)
    new_cwd = os.path.join(base_cwd, customer, dev_dir, "models")
    os.chdir(new_cwd)

def change_cwd_file_dev(customer):
    #cwd = os.getcwd()
    dev_dir = save_models.get_dev_dir(customer)
    new_cwd = os.path.join(base_cwd, customer, dev_dir, "files")
    os.chdir(new_cwd)
    
def fetch_training_data(sum_col,priority_col,customer):
    # Hardcoaded customer table name, we will need to create a table or dictionary to map customers to tables.
    customer_tbl = customer + "_ticketdata"   
    params = config()
    conn = psycopg2.connect(**params)
    print("PostgreSQL Connection object ",conn)
    cur = conn.cursor()
    training_data = pd.read_sql_query('SELECT %s,%s FROM %s'%(sum_col,priority_col,customer_tbl),conn)
    cur.close()
    conn.close()     
    return training_data

def sanity_check(training_data,sum_col,priority_col):
    training_data = training_data[[sum_col, priority_col]]
    training_data = training_data.dropna()
    training_data = training_data[:-1]
    value_counts = training_data[priority_col].value_counts()
    return training_data


def label_encode(training_data,sum_col,priority_col):
    le = preprocessing.LabelEncoder()
    le.fit(training_data[priority_col])
    print("-------------------------------------------------------------------------")
    print(le.classes_)
    original_priorities = le.classes_
    original_priorities = original_priorities.tolist()
    length_priorities = len(original_priorities)
    #print(type(original_priorities))
    #print(original_priorities)
    print(np.unique(le.transform(training_data[priority_col])))
    transf_priorities = np.unique(le.transform(training_data[priority_col]))
    #print(type(transf_priorities))
    transf_priorities = transf_priorities.tolist()
    training_data[priority_col] = le.transform(training_data[priority_col])
    le2 = preprocessing.LabelEncoder()
    le2.fit(training_data[sum_col])
    label_file = 'label_encode_col.sav'
    pickle.dump(le2, open(label_file, 'wb'))
    #training_data["sum_col"] = le2.transform(training_data[sum_col])
    return training_data,original_priorities,transf_priorities,length_priorities


def split_target(training_data,sum_col,priority_col):
    value_counts = training_data[priority_col].value_counts()
    x = training_data[sum_col]
    #X_not_labled = training_data[sum_col]
    y = training_data[priority_col]
    return x, y


def train_validation_split(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=2, stratify=y)
    pkl_filename = "label_encode_col.sav"
    file = open(pkl_filename, 'rb')
    label = pickle.load(file)
    file.close()
    x_train = label.transform(X_train)
    x_test = label.transform(X_test)
    return x_train, x_test, y_train, y_test, X_train, X_test


def tf_model_train(x_train, y_train, x_test, y_test,length_priorities,model_epoch,model_neurons,customer):
    X_train = x_train.astype(np.uint8)
    X_test = x_test.astype(np.uint8)
    Y_train = to_categorical(y_train, length_priorities)
    Y_test = to_categorical(y_test, length_priorities)
    model = Sequential()
    model.add(Dense(model_neurons, activation='relu'))
    model.add(Dense(model_neurons, activation='relu'))
    model.add(Dense(model_neurons, activation='relu'))
    model.add(Dense(model_neurons, activation='relu'))
    model.add(Dense(length_priorities, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
    model.fit(X_train, Y_train, epochs=model_epoch, batch_size=150, callbacks=[early_stop])
    model.save(os.getcwd())
    y_pred = model.predict_classes(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    Accuracy_list.append(accuracy)
    Model_list.append("saved_model.pb")
    return model,accuracy
    
def sgd_model_train(X_train, X_test, Y_train, Y_test,customer):
    # for applying operations sequentially
    sgd_model = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(random_state=42, max_iter=200, tol=None)),
                    ])
   
    sgd_model.fit(X_train, Y_train)
  
    filename = 'sgd_model.pb'
    pickle.dump(sgd_model, open(filename, 'wb'))
    y_pred = sgd_model.predict(X_test)
    sgd_accuracy = accuracy_score(Y_test, y_pred)
    Accuracy_list.append(sgd_accuracy)
    Model_list.append("sgd_model.pb")
    return sgd_model,sgd_accuracy     

 #Threads   
class myThread (threading.Thread):
    def __init__(self,name, X_train, X_test, y_train, y_test,customer,x_train,x_test,length_priorities,model_epoch,model_neurons):
        threading.Thread.__init__(self)
        self.name = name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.customer = customer
        self.x_train = x_train
        self.x_test = x_test
        self.length_priorities = length_priorities
        self.model_epoch = model_epoch
        self.model_neurons = model_neurons
    def run(self):
        print("--------------------Starting model training for: " + self.name)
        if self.name == "Model_SGD":
            SGD_model,SGD_accuracy = sgd_model_train(self.X_train,self.X_test, self.y_train, self.y_test,self.customer)
        elif self.name == "Model_TF":
            TF_model,TF_accuracy = tf_model_train(self.x_train, self.y_train, self.x_test, self.y_test,self.length_priorities,self.model_epoch,self.model_neurons,self.customer)
        else:
            print("Pass")
 
        print("-------------------Training completed for: " + self.name)
        


def initate_process(customer,sum_col,priority_col,model_epoch,model_neurons,algo_lst):
    cur_dir = save_models.model_archive(customer)
    training_data = fetch_training_data(sum_col,priority_col,customer)
    training_data = sanity_check(training_data,sum_col,priority_col)
    training_data[sum_col] = training_data[sum_col].apply(text_processing)
    change_cwd_model(customer,cur_dir)
    print("------------------------", training_data[sum_col].iloc[0])
    training_data,original_priorities,transf_priorities,length_priorities = label_encode(training_data,sum_col,priority_col)
    x, y = split_target(training_data,sum_col,priority_col)
    x_train, x_test, y_train, y_test, X_train, X_test = train_validation_split(x, y)
    
#Call to Threads

    for algo in algo_lst:

        thread = myThread(algo, X_train, X_test, y_train, y_test,customer,x_train,x_test,length_priorities,model_epoch,model_neurons)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()
    print("---------------Exiting Model training, all models trained")  
#call ends

    df = pd.DataFrame() 
    df['Original'] = original_priorities
    df['Transformed'] = transf_priorities
    print(df)
    change_cwd_file(customer,cur_dir)
    df.to_csv("label_output.csv") 
    Accuracy_df = pd.DataFrame(columns=["Model","Accuracy"])
    print("Accuracy Comparison")
    Accuracy_df["Model"] = Model_list
    Accuracy_df["Accuracy"] = Accuracy_list
    Accuracy_df = Accuracy_df.sort_values(by='Accuracy', ascending=False) 
    Accuracy_df.to_csv("priority_output.csv")
    best_accuracy = Accuracy_df['Accuracy'].iloc[0]
    print(Accuracy_df)
    msg = "Priority model training completed for customer: " + customer
    accuracy_dict = {"msg":msg,"accuracy":best_accuracy}
    best_accuracy = str(best_accuracy)
    best_accuracy = best_accuracy[:7]
    mt.set_accuracy(customer,best_accuracy)
    Accuracy_df = Accuracy_df.iloc[0:0]
    Accuracy_list.clear()
    Model_list.clear()
    os.chdir(base_dir)
   
    return accuracy_dict
    

def detect_priority(customer,summary,env):
    if env == "prod":
        change_cwd_file_prod(customer)
    else:
        change_cwd_file_dev(customer)
        
    output_df = pd.read_csv("priority_output.csv")
    if env == "prod":
        change_cwd_model_prod(customer)
    else:
        change_cwd_model_dev(customer)
    print("-------",os.getcwd())
    
    filename2 = output_df['Model'].iloc[0]
    if filename2 == "saved_model.pb":
        loaded_model = keras.models.load_model(os.getcwd())

        pkl_filename = "label_encode_col.sav"
        file = open(pkl_filename, 'rb')
        label = pickle.load(file)
        file.close()
        
        pred_df = pd.DataFrame(columns=["col"])
        pred_df.loc[len(pred_df.index), 'col'] = summary       
        pred_df["col"] = pred_df["col"].apply(text_processing)
        pred_df["col"] = label.transform(pred_df["col"])
        priority = loaded_model.predict_classes(pred_df["col"])
        print(priority)
    
    else:
        loaded_model = pickle.load(open(filename2, 'rb'))
        new_doc = summary
        new_doc = [new_doc]
        print(type(new_doc))
        print(new_doc)
        priority = loaded_model.predict(new_doc)
    
    return priority
