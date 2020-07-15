import paho.mqtt.client as mqtt
import json
import csv
import pandas as pd
import os
import ast

subscribe = "ENTC"

def on_connect(client, userdata, flags, rc):
    if (rc==0):
        print ("Connection Successfull!")
        client.subscribe(subscribe)
        print ("Subscribe to %s"%subscribe)
    else:
        print("Connected with result code "+str(rc))

        
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    data_store_csv(msg.payload)

    
    
def data_store_csv(data1):

    # Define result array
    result = {"1e:60:24:46:7f:ee": -90, "20:8d:d9:77:b3:56":-90,"41:97:7c:67:3d:88":-90}
    # Define csv file path
    filename = "ble_data.csv"

    try:
        # Get data as this format : {"id" : "1","UoM_Wireless1" : "-58","eduroam1" : "-89","UoM_Wireless1" : "-89","eduroam1" : "-57"}
        data = ast.literal_eval(data1.decode("utf-8"))
        print(data)

        for key,value in data.items():
            if key in result and result[key] == -100:
                result[key] = float(value)
        
        
        #result["id"] = int(data["id"])

        # Create one row of csv file
        df = pd.Series(result).to_frame().T

        print(df)

        # Write to csv file
        df.to_csv(filename,index=False,mode='a',header=(not os.path.exists(filename)))
        
    except Exception as e:
        print(e)
