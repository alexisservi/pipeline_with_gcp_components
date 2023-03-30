from kfp.registry import RegistryClient
PROJECT_ID = "almacafe-ml-poc"
client = RegistryClient(host="https://us-central1-kfp.pkg.dev/{}/ml-automation-kfp-repo".format(PROJECT_ID))


#V1 Compiler -> it works...!
templateName, versionName = client.upload_pipeline( 
  file_name="gcp_components_pipeline.yaml",
  tags=["v1", "latest"],
  extra_headers={"description":"This is an example pipeline template. Alexis"})


"""
templateName, versionName = client.upload_pipeline(
  file_name="hello_world_pipeline.json",
  tags=["v2", "latest"],
  extra_headers={"description":"This is an example pipeline template with V2 compiler. Alexis"}) 
"""