import os
import sys
import json
import joblib
import argparse
import traceback

import shutil

from dotenv import load_dotenv

from azureml.core import ScriptRunConfig
from azureml.core.environment import Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Run, Experiment, Workspace, Dataset, Datastore

# For local development, set values in this section
load_dotenv()


# --- METHODS --------------------------------------------------------------------------------
def prepareMachines(ws):
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    compute_min_nodes = int(os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES"))
    compute_max_nodes = int(os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES"))
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU")

    # connect to compute target if it exists
    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print("Found compute target, will use this one: " + compute_name)
    # else create new compute target
    else:
        print("Creating new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size, min_nodes = compute_min_nodes, max_nodes = compute_max_nodes)
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    return compute_target


def prepareEnv(ws, env_name):
    # create environment & add package dependencies
    env = Environment(env_name)
    cd = CondaDependencies.create( pip_packages=['azureml-dataset-runtime[pandas,fuse]', 
                                                    'azureml-defaults', 
                                                    'scikit-learn', 
                                                    'tensorflow'])
    env.python.conda_dependencies = cd

    # Register environment to re-use later
    env.register(workspace=ws)

    return env


def prepareTraining(dataset, script_folder, compute_target, env):
    # get environment variables
    train_script_name = os.environ.get('TRAIN_SCRIPT_NAME')
    regularization = float(os.environ.get('REGULARIZATION'))
    model_name = os.environ.get("MODEL_NAME")

    # define arguments and training script
    args = ['--data-folder', dataset.as_mount(), 
            '--regularization', regularization,
            '--model_name', model_name]

    # create a job to run
    src = ScriptRunConfig(source_directory=script_folder, 
                            script=train_script_name, 
                            arguments=args, 
                            compute_target=compute_target, 
                            environment=env)
    
    return src


# --- MAIN --------------------------------------------------------------------------------
def main():
    # ===========================================
    print('Executing - 02_Training')
    # authentication
    cli_auth = AzureCliAuthentication()

    # get environment variables 
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group = os.environ.get("RESOURCE_GROUP")
    workspace_name = os.environ.get("WORKSPACE_NAME")

    experiment_name = os.environ.get("EXPERIMENT_NAME")
    env_name = os.environ.get("AML_ENV_NAME")
    dataset_name = os.environ.get('DATASET_NAME')
    temp_state_directory = os.environ.get('TEMP_STATE_DIRECTORY')

    #create path to scripts inside root directory of steps
    root_dir = os.environ.get('ROOT_DIR')
    script_folder = os.path.join(root_dir, 'scripts')

    # setup workspace + datastore
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )
    print(f'Using dataset: {dataset_name}')

    # Prepare dataset, compute target, environment and job
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
    compute_target = prepareMachines(ws)
    env = prepareEnv(ws, env_name)
    src = prepareTraining(dataset, script_folder, compute_target, env)

    ## Start training
    exp = Experiment(workspace=ws, name=experiment_name)
    run = exp.submit(config=src)
    run.wait_for_completion()
    #cant serialize inputDatasets, outputDatasets
    run_details = {k:v for k,v in run.get_details().items() if k not in ['inputDatasets', 'outputDatasets']}

    # save results/information in temporary directory
    path_json = os.path.join(temp_state_directory, 'training_run.json')
    with open(path_json, 'w') as training_run_json:
        json.dump(run_details, training_run_json)

    print('Executing - 02_Training - SUCCES')
    # ===========================================

if __name__ == '__main__':
    main()
