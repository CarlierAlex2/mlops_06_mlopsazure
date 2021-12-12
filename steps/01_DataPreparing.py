import argparse
import json
import os
import sys
import traceback
import math
from glob import glob

import joblib
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from azureml.core import Dataset, Datastore, Experiment, Run, Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.opendatasets import MNIST

# For local development, set values in this section
load_dotenv() #add values to.env file and insert values into script


# --- METHODS --------------------------------------------------------------------------------
def downloadDataMNIST(data_folder, ws): # download our data
    # get environment variables
    dataset_name = os.environ.get('DATASET_NAME')
    dataset_description = os.environ.get('DATASET_DESCRIPTION')
    dataset_new_version = os.environ.get('DATASET_NEW_VERSION') == 'true'# string to bool

    # get + register MNIST dataset
    mnist_file_dataset = MNIST.get_file_dataset()
    mnist_file_dataset.download(data_folder, overwrite=True)
    mnist_file_dataset = mnist_file_dataset.register(workspace=ws, 
                                                    name=dataset_name,
                                                    description=dataset_description,
                                                    create_new_version=dataset_new_version)
    #return mnist_file_dataset
    return {'name' : dataset_name, 'description' : dataset_description}


# --- MAIN --------------------------------------------------------------------------------
def main():
    # ===========================================
    print('Executing - 01_DataPreparing')
    # authentication
    cli_auth = AzureCliAuthentication()

    # get environment variables # get instead of [], default None
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group = os.environ.get("RESOURCE_GROUP")
    workspace_name = os.environ.get("WORKSPACE_NAME")

    data_folder = os.environ.get('DATA_FOLDER')
    temp_state_directory = os.environ.get('TEMP_STATE_DIRECTORY')

    # connect to workspace + datastore
    # datastore is where all of our datasets will be uploaded and stored
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )
    datastore = Datastore(ws)

    # data
    data_folder = os.path.join(os.getcwd(), data_folder) #absolute path
    os.makedirs(data_folder, exist_ok=True) # add directory if doesn't exist
    mnist_file_dataset = downloadDataMNIST(data_folder, ws)

    # save results/information in temporary directory
    os.makedirs(temp_state_directory, exist_ok=True)
    path_json = os.path.join(temp_state_directory, 'dataset.json')
    with open(path_json, 'w') as dataset_json:
        json.dump(mnist_file_dataset, dataset_json)

    # ===========================================
    print('Executing - 01_DataPreparing - SUCCES')

if __name__ == '__main__':
    main()
