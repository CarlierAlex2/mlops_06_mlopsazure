import os
import sys
import json
import uuid
import joblib
import argparse
import traceback
from dotenv import load_dotenv

from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Run, Experiment, Workspace, Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import Webservice, AciWebservice
from azureml.core.authentication import AzureCliAuthentication

# For local development, set values in this section
load_dotenv()


# --- METHODS --------------------------------------------------------------------------------
def getConfiguration(details_file):
    try:
        with open(details_file) as f:
            config = json.load(f)
    except Exception as e:
        print(e)
        sys.exit(0)
    return config


# --- MAIN --------------------------------------------------------------------------------
def main():
    # ===========================================
    print('Executing - 04_DeployModel')
    # authentication
    cli_auth = AzureCliAuthentication()

    # get environment variables
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group = os.environ.get("RESOURCE_GROUP")
    workspace_name = os.environ.get("WORKSPACE_NAME")

    env_name = os.environ.get("AML_ENV_NAME")
    model_name = os.environ.get("MODEL_NAME")

    temp_state_directory = os.environ.get('TEMP_STATE_DIRECTORY')
    root_dir = os.environ.get('ROOT_DIR')
    score_script_path = os.path.join(root_dir, 'scripts', 'score.py')

    # connect to workspace
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )
    model = Model(ws, model_name)

    # get model from configuration
    #path_json = os.path.join(temp_state_directory, 'model_details.json')
    #config = getConfiguration(path_json)
    #model = Model.deserialize(workspace=ws, model_payload=config['model'])
    env = Environment.get(workspace=ws, name=env_name, version="1")
    #cd = CondaDependencies.create( pip_packages=['azureml-defaults','numpy', 'tensorflow'])

    # create environment & add package dependencies & register
    #env.python.conda_dependencies = cd
    #env.register(workspace=ws)
    inference_config = InferenceConfig(entry_script=score_script_path, environment=env)

    # Azure Container Instance runs Docker container on Azure
    aciconfig = AciWebservice.deploy_configuration(
        cpu_cores=1, 
        memory_gb=1, 
        tags={"data": "MNIST",  "method" : "sklearn"}, 
        description='Predict MNIST with sklearn'
    )

    # deploy service
    service_name = 'mnist-digits-svc-' + str(uuid.uuid4())[:4]
    service = Model.deploy(workspace=ws, 
                        name=service_name, 
                        models=[model], 
                        inference_config=inference_config, 
                        deployment_config=aciconfig)
    service.wait_for_deployment(show_output=True)
    print(service.scoring_uri)

    # save results/information in temporary directory
    path_json = os.path.join(temp_state_directory, 'service_details.json')
    with open(path_json, "w") as service_details:
        json.dump(service.serialize(), service_details)
    
    # ===========================================
    print('Executing - 04_DeployModel - SUCCES')

if __name__ == '__main__':
    main()