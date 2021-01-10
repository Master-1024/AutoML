import mysql.connector as mysql
import pandas as pd

mydb = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "root",
    database="mydatabase",
  auth_plugin='mysql_native_password'
)

def set_dev_model(customer,modeldate):
    try:
        mycursor = mydb.cursor()
        sql = "UPDATE models4 SET flag = 'arch' WHERE flag = 'dev' and customer = %s"
        adr = (customer, )
        mycursor.execute(sql, adr)
        mydb.commit()
        
        sql = "UPDATE models4 SET flag = 'dev' WHERE datecreated = %s and customer = %s"
        adr = (modeldate,customer, )
        mycursor.execute(sql, adr)
        mydb.commit()
        return "Model is set as Dev"
    except Exception as e:
        print(e)
        return "Error in transaction"
        
def set_prod_model(customer,modeldate):
    try:
        mycursor = mydb.cursor()
        
        sql = "UPDATE models4 SET flag = 'arch' WHERE flag = 'prod' and customer = %s"
        adr = (customer, )
        mycursor.execute(sql, adr)
        mydb.commit()
        
        sql = "UPDATE models4 SET flag = 'prod' WHERE datecreated = %s and customer = %s"
        adr = (modeldate,customer, )
        mycursor.execute(sql, adr)
        mydb.commit()
        return "Model is set as Prod"
    except Exception as e:
        print(e)
        return "Error in transaction"
        
def set_arch_model(customer,modeldate):
    try:
        mycursor = mydb.cursor()
        sql = "UPDATE models4 SET flag = 'arch' WHERE datecreated = %s and customer = %s"
        adr = (modeldate,customer, )
        mycursor.execute(sql, adr)
        mydb.commit()
        return "Model is set as Arch"
    except Exception as e:
        print(e)
        return "Error in transaction"
        
def get_model_details(customer):
    try:
        mycursor = mydb.cursor()
        customer = str(customer)
        training_data = pd.read_sql('SELECT * FROM models4 WHERE customer = '+"'"+customer+"'", con=mydb)
        return training_data
    except Exception as e:
        print(e)
        temp = pd.DataFrame()
        return temp
        
def set_accuracy(customer,accuracy):
    try:
        mycursor = mydb.cursor()
        sql = "UPDATE models4 SET accuracy = %s WHERE accuracy = 'NA' and customer = %s"
        adr = (accuracy,customer, )
        mycursor.execute(sql, adr)
        mydb.commit()
        return "Model is set as Arch"
    except Exception as e:
        print(e)
        return "Error in transaction"