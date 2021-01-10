import mysql.connector as mysql
import datetime
import os
import pathlib

def model_archive(customer): 
    try:
        mydb = mysql.connect(
        host="localhost",
        user="root",
        passwd="root",
        database="mydatabase",
        auth_plugin='mysql_native_password'
        )
        print(mydb)   
        mycursor = mydb.cursor()
        print(mycursor)
        model_date = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M')
        print(model_date)
        model_flag = "arch"
        sql = "INSERT INTO models4 (customer, modelname, datecreated, flag, accuracy) VALUES (%s, %s, %s, %s, %s)"
        val = (customer, customer+"_model_"+model_date, model_date, model_flag, 'NA')
        mycursor.execute(sql, val)
        print("here")
        mydb.commit()
        print(mycursor.rowcount, "record inserted.")
        new_dir = customer+"_model_"+model_date
        create_customer_modeldir(customer,new_dir)
        return new_dir
    except Exception as e:
        print(e)
        return "error"
    
def get_dev_dir(customer):
    try:
        mydb = mysql.connect(
        host="localhost",
        user="root",
        passwd="root",
        database="mydatabase",
        auth_plugin='mysql_native_password'
        )   
        mycursor = mydb.cursor()
        sql = "SELECT modelname FROM models4 WHERE flag = %s and customer = %s"
        adr = ("dev",customer, )
        mycursor.execute(sql, adr)
        myresult = mycursor.fetchall()
        for x in myresult:
            model_date = x
        output = str(model_date)[2:-3]
        print("-----------",output)
        return output
    except Exception as e:
        print(e)
        return "no path"    

def get_prod_dir(customer):
    try:
        mydb = mysql.connect(
        host="localhost",
        user="root",
        passwd="root",
        database="mydatabase",
        auth_plugin='mysql_native_password'
        )   
        mycursor = mydb.cursor()
        sql = "SELECT modelname FROM models4 WHERE flag = %s and customer = %s"
        adr = ("prod",customer, )
        mycursor.execute(sql, adr)
        myresult = mycursor.fetchall()
        for x in myresult:
            model_date = x
        output = str(model_date)[2:-3]
        print("-----------",output)
        return output
    except Exception as e:
        print(e)
        return "no path" 
   
def create_customer_modeldir(customer,new_dir):
    base_dir = pathlib.Path(__file__).parent.absolute()
    new_cwd = os.path.join(base_dir, customer, new_dir)
    os.mkdir(new_cwd)
    new_model = os.path.join(new_cwd, "models")
    new_file = os.path.join(new_cwd, "files")
    os.mkdir(new_model)
    os.mkdir(new_file)
    return "done"
    
