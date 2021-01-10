from typing import Optional
from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from timeit import default_timer as timer
import os
import classifier as pc
import model_transactions as mt
import pandas as pd
import json

import pathlib


class Item(BaseModel):
    #name: str
    #eid: Optional[int] = None
    customer: str
    summary: Optional[str] = "None"
    col_name_summary: Optional[str] = "short_description"
    col_name_priority: Optional[str] = "priority"
    epochs_no: Optional[int] = 20
    neurons_no: Optional[int] = 50
    algo_list: Optional[list]= []
    env: Optional[str] = "dev"
    modeldate: Optional[str] = "NaN"

app = FastAPI()

base_dir = pathlib.Path(__file__).parent.absolute()
os.chdir(base_dir)
base_cwd = os.getcwd()
 
@app.post("/train_priority/")
async def train_priority(item: Item):
        cust = item.customer
        sum_col = item.col_name_summary
        priority_col = item.col_name_priority
        model_epoch = item.epochs_no
        model_neurons = item.neurons_no
        algo_lst = item.algo_list
        if not algo_lst:
            #output = {"Error":"No Algorithms selected"}
            #return output
            algo_lst = ["Model_SGD","Model_TF"]
        try:
            tic=timer()
            accuracy_dict = pc.initate_process(cust,sum_col,priority_col,model_epoch,model_neurons,algo_lst)
            toc=timer()
            output = accuracy_dict
            total = toc - tic
            print("Total time for training ",total)
            #output = json.dumps(output, indent = 4)
        except Exception as e:
            print(e)
            output = {"Error":"Error in training"}
            #output = json.dumps(output, indent = 4)
        return output
 
@app.post("/get_priority/")
async def get_priority(item: Item, response: Response):
        cust = item.customer
        summ = item.summary
        env = item.env
        if summ == "None":
            output = {"Error":"Summary/ticket not present"}
            #output = json.dumps(output, indent = 4)
            response.status_code = status.HTTP_400_BAD_REQUEST
            return output
        try:
            ticket_priority = pc.detect_priority(cust,summ,env)
            ticket_priority = str(ticket_priority)
            ticket_priority = ticket_priority[1:-1]
            ticket_priority = int(ticket_priority)
            if env == "dev":
                pc.change_cwd_file_dev(cust)
            else:
                pc.change_cwd_file_prod(cust)
            priority_df = pd.read_csv("label_output.csv")
            
            outdict = priority_df.set_index('Transformed')['Original'].to_dict()
            print("--",type(outdict),outdict)
           
            prio = outdict.get(ticket_priority)
            print(prio)
            output = {"Priority":prio}
        except Exception as e:
            print(e)
            err = str(e)
            if "[WinError 3] The system cannot find the path specified" in err:
                error = env + " model is not configured, please set a model as a " + env +" model"
                output = {"Error": error}
                response.status_code = status.HTTP_400_BAD_REQUEST
            else:
                output = {"Error" : "Error in fetching priority"}
                response.status_code = status.HTTP_400_BAD_REQUEST
        #output = json.dumps(output, indent = 4) 
        return output
     
@app.post("/set_model_to_dev/")
async def set_model_to_dev(item: Item):
    cust = item.customer
    modeldate = item.modeldate
    try:
        out = mt.set_dev_model(cust,modeldate)
        output = {"msg":out}
    except Exception as e:
        print(e)
        output = {"Error": "Error in transaction"}
    #output = json.dumps(output, indent = 4) 
    return output 
    
@app.post("/set_model_to_prod/")
async def set_model_to_prod(item: Item):
    cust = item.customer
    modeldate = item.modeldate
    try:
        out = mt.set_prod_model(cust,modeldate)
        output = {"msg":out}
    except Exception as e:
        print(e)
        output = {"Error": "Error in transaction"}
    #output = json.dumps(output, indent = 4) 
    return output 

@app.post("/set_model_to_arch/")
async def set_model_to_arch(item: Item):
    cust = item.customer
    modeldate = item.modeldate
    try:
        out = mt.set_arch_model(cust,modeldate)
        output = {"msg":out}
    except Exception as e:
        print(e)
        output = {"Error": "Error in transaction"}
    #output = json.dumps(output, indent = 4) 
    return output 
    
@app.post("/get_model_details/")
async def get_model_details(item: Item):
    cust = item.customer
    try:
        out = mt.get_model_details(cust)
        #output = {"msg":out}
        
        result = out.to_json(orient="index")
        parsed = json.loads(result)
        #output = json.dumps(parsed, indent=4)
        
    except Exception as e:
        print(e)
        output = {"Error": "Error in transaction"}
    #output = json.dumps(output, indent = 4) 
    return parsed 
 