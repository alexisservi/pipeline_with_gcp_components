# CREARTE AND COMPILE A PIPELINE
import os
import kfp
from kfp import dsl
from kfp import compiler
#import kfp.components as comp
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    InputPath, OutputPath, )
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from typing import NamedTuple

from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from kfp.v2.components import importer_node

import google.cloud.aiplatform as aip

PROJECT_ID = "almacafe-ml-poc"
BUCKET_URI = "gs://ml-auto-pipelines-bucket"
PIPELINE_ROOT = "{}/pipeline_root/bikes_weather".format(BUCKET_URI)


hp_dict: str = '{"num_hidden_layers": 3, "hidden_size": 32, "learning_rate": 0.01, "epochs": 1, "steps_per_epoch": -1}'
data_dir: str = "gs://aju-dev-demos-codelabs/bikes_weather/"
TRAINER_ARGS = ["--data-dir", data_dir, "--hptune-dict", hp_dict]


# create working dir to pass to job spec
WORKING_DIR = f"{PIPELINE_ROOT}" #/{UUID}"

MODEL_DISPLAY_NAME = f"train_deploy_with_gcp_components"
print(TRAINER_ARGS, WORKING_DIR, MODEL_DISPLAY_NAME)

# Initialize Vertex AI SDK for Python
aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

#---------------------------------------------------------------------------------------------------
@dsl.pipeline(name='custom-components-v1', description='A pipeline with custom components')
def gcp_components_pipeline(
    project: str = PROJECT_ID,
    model_display_name: str = MODEL_DISPLAY_NAME,
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
    ):


    custom_job_task = CustomTrainingJobOp(
        project=project,
        display_name="model-training",
        worker_pool_specs=[
            {
                "containerSpec": {
                    "args": TRAINER_ARGS,
                    "env": [{"name": "AIP_MODEL_DIR", "value": WORKING_DIR}],
                    "imageUri": "gcr.io/google-samples/bw-cc-train:latest",
                },
                "replicaCount": "1",
                "machineSpec": {
                    "machineType": "n1-standard-4",
                    "accelerator_type": aip.gapic.AcceleratorType.NVIDIA_TESLA_K80,
                    "accelerator_count": 1,
                },
            }
        ],
    )

    import_unmanaged_model_task = importer_node.importer(
        artifact_uri=WORKING_DIR,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
            },
        },
    ).after(custom_job_task)

    # Upload model if it was created
    with dsl.Condition(import_unmanaged_model_task != None):
        # Upload to Vertex
        model_upload_op = ModelUploadOp(
            project=project,
            display_name=model_display_name,
            unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
        )
        model_upload_op.after(import_unmanaged_model_task)

        # Create endpoint
        endpoint_create_op = EndpointCreateOp(
            project=project,
            display_name="pipelines-created-endpoint",
        )

        # Deploy model to endpoint
        ModelDeployOp(
            endpoint=endpoint_create_op.outputs["endpoint"],
            model=model_upload_op.outputs["model"],
            deployed_model_display_name=model_display_name,
            dedicated_resources_machine_type="n1-standard-4",
            dedicated_resources_min_replica_count=1,
            dedicated_resources_max_replica_count=1,
        )
    
#------------------------------------------
# Compile pipeline
# V1 Compiler -> it works... 
compiler.Compiler().compile(
    pipeline_func=gcp_components_pipeline,
    package_path='gcp_components_pipeline.yaml', 
    #type_check=False
    )

print("List directory files")
print(os.listdir())
