apt update
apt install python3 python3-dev python3-venv -y


apt-get install wget -y

echo "-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "VERIFICANDO VERSION PYTHON"
python3 -V

wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py

mkdir env_folder
cd env_folder
python -m venv env
source env/bin/activate
cd ..
#pip install protobuf==3.19.6
#pip install google-cloud-aiplatform  --upgrade
#pip install --pre kfp --upgrade
pip install google-cloud-aiplatform
pip install --pre kfp

#pip install --upgrade google-cloud-pipeline-components
pip install --pre  google-cloud-pipeline-components

# Configure GCP permisions
#https://cloud.google.com/vertex-ai/docs/pipelines/create-pipeline-template#configuring_permissions

#----------------------------------------------
# Create and compile pipeline
python3 pl_gcp_components.py

#----------------------------------------------
# Upload template
python3 upload_template.py
