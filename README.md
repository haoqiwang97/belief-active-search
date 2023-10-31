# belief-active-search

## Folders and files
- `data`
    - `best_embedding_2023-08-31.csv` from `belief/MDS_analysis/perceptual_maps`
    - `img_num_map.csv` from `belief/belief/trial_design`
    - `20220801_process_checkpoint_ps.csv` from `belief/img_selection`
    - `breast_states_clusters.csv` from `belief/img_selection`
- `database`
    - `belief.db`
- `img_database_2d`, 100 images
- `templates`, html pages
    - `layout.html`, base for other pages
    - `trial.html`, display 2 images
- `temporary`, temporary files and will be deleted
- `main.py`, display webpage
- `config.py`, stores paths, credentials
- `create_db.py`, create database and tables
- `drop_db.py`, delete tables
- `requirements.txt`, `pip freeze > requirements.txt `

## Commands
Local test
```zsh
pip list --format=freeze > requirements_pip.txt  

uvicorn main:app --reload

docker build -t belief .
docker run -dp 80:80 belief

# use volume
docker volume create belief-db
docker run -dp 80:80 -v belief-db:/app/database belief
```

Azure, deploy container and mount volume, first manually upload database file (command line written below), https://learn.microsoft.com/en-us/azure/container-instances/container-instances-volume-azure-files

```zsh
# sign in BMIL
# az login -u utbmil@gmail.com -p xx, I do not know the password
az login # interactive login

ACI_PERS_RESOURCE_GROUP=beliefResourceGroup
ACI_PERS_STORAGE_ACCOUNT_NAME=utbeliefstorageaccount
ACI_PERS_LOCATION=eastus
ACI_PERS_SHARE_NAME=beliefshare

# Create resource group
az group create --name $ACI_PERS_RESOURCE_GROUP --location $ACI_PERS_LOCATION

# Create an Azure Container Registry
# previously I used ACR_NAME=bmilbelief, so bmilbelief.azurecr.io is already in use
ACR_NAME=utbmilbelief
az acr create --resource-group $ACI_PERS_RESOURCE_GROUP --name $ACR_NAME --sku Basic

# log in ACR
az acr login --name $ACR_NAME
az acr show --name $ACR_NAME --query loginServer --output table

# Tag container image
acrLoginServer=utbmilbelief.azurecr.io
# acrLoginServer=$(az acr show --name $ACR_NAME --query loginServer)
docker tag belief $acrLoginServer/belief:v1

# Push image to ACR (Azure Container Registry)
docker push $acrLoginServer/belief:v1

# List images in Azure Container Registry
az acr repository list --name $ACR_NAME --output table

# Create the storage account with the parameters
az storage account create \
    --resource-group $ACI_PERS_RESOURCE_GROUP \
    --name $ACI_PERS_STORAGE_ACCOUNT_NAME \
    --location $ACI_PERS_LOCATION \
    --sku Standard_LRS

# Create the file share
az storage share create \
  --name $ACI_PERS_SHARE_NAME \
  --account-name $ACI_PERS_STORAGE_ACCOUNT_NAME

# Storage account key
STORAGE_KEY=$(az storage account keys list --resource-group $ACI_PERS_RESOURCE_GROUP --account-name $ACI_PERS_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)
echo $STORAGE_KEY

# Manually upload database file
az storage file upload \
    --account-name $ACI_PERS_STORAGE_ACCOUNT_NAME \
    --share-name $ACI_PERS_SHARE_NAME \
    --source "/Users/haoqiwang/Documents/belief-active-search/database/belief.db" \
    --path "belief.db"

# Deploy container and mount volume
CONTAINER_NAME=bmil-belief-app

# remember to allow access key and update password
# path, beliefResourceGroup>utbmilbelief
# get password, #utbmilbelief 
az acr credential show --name $ACR_NAME
REGISTRY_PASSWORD=IlsbNa0DW8+XOGSwYZub9ETUjPOXKY7Asg3caYLw7D+ACRBj+a3B

az container create \
    --resource-group $ACI_PERS_RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image utbmilbelief.azurecr.io/belief:v1 \
    --dns-name-label utbmilbelief \
    --ports 80 \
    --azure-file-volume-account-name $ACI_PERS_STORAGE_ACCOUNT_NAME \
    --azure-file-volume-account-key $STORAGE_KEY \
    --azure-file-volume-share-name $ACI_PERS_SHARE_NAME \
    --azure-file-volume-mount-path /app/database/ \
    --registry-login-server $acrLoginServer --registry-username utbmilbelief --registry-password $REGISTRY_PASSWORD

# update container, note v1 -> v4
# need 2 cpu, otherwise it cannot run, stan is overkill here, future may use algo that needs less cpu
az container create \
    --resource-group $ACI_PERS_RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image utbmilbelief.azurecr.io/belief:v4 \
    --dns-name-label utbmilbelief \
    --ports 80 \
    --azure-file-volume-account-name $ACI_PERS_STORAGE_ACCOUNT_NAME \
    --azure-file-volume-account-key $STORAGE_KEY \
    --azure-file-volume-share-name $ACI_PERS_SHARE_NAME \
    --azure-file-volume-mount-path /app/database/ \
    --registry-login-server $acrLoginServer --registry-username utbmilbelief --registry-password $REGISTRY_PASSWORD \
    --cpu 2 --memory 3.5

# View the application and container logs
az container show --resource-group $ACI_PERS_RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn
az container logs --resource-group $ACI_PERS_RESOURCE_GROUP --name $CONTAINER_NAME

# website: utbmilbelief.eastus.azurecontainer.io

# run command in container instance
az container exec --resource-group $ACI_PERS_RESOURCE_GROUP --name $CONTAINER_NAME --exec-command "gcc --version"

# stop container
az container stop --name $CONTAINER_NAME \
                  --resource-group $ACI_PERS_RESOURCE_GROUP

# download database
DEST="/Users/haoqiwang/Downloads/belief.db"
# DEST="/Users/haoqiwang/Documents/belief-active-search/database/belief.db"
az storage file download \
    --account-name $ACI_PERS_STORAGE_ACCOUNT_NAME \
    --share-name $ACI_PERS_SHARE_NAME \
    --path "belief.db" \
    --dest $DEST \
    --output none

#####################
# delete everything #
#####################
az container delete --resource-group $ACI_PERS_RESOURCE_GROUP --name $CONTAINER_NAME
az group delete --name $ACI_PERS_RESOURCE_GROUP
```

Difficulties
- Use Azure cloud, store database
- Rewrite backend algorithms
- Compile pystan in container

# Old
Prepare pieces
- local, local with volume, cloud
- simple database, can add row, display, mount to volume
- simple show image

1. create new environment, combine stan_env & fastapi
    1. python
1. GUI of pair comparison
    1. template
    1. image
1. database

Azure, Deploy container without mounting volume, https://learn.microsoft.com/en-us/azure/container-instances/container-instances-tutorial-prepare-acr

```zsh
# Change these four parameters as needed
ACI_PERS_RESOURCE_GROUP=beliefResourceGroup
ACI_PERS_STORAGE_ACCOUNT_NAME=beliefstorageaccount
ACI_PERS_LOCATION=eastus
ACI_PERS_SHARE_NAME=beliefshare

# Create resource group
az group create --name $ACI_PERS_RESOURCE_GROUP --location eastus

# Create an Azure container registry
ACR_NAME=utbmilbelief
az acr create --resource-group $ACI_PERS_RESOURCE_GROUP --name $ACR_NAME --sku Basic

az acr login --name $ACR_NAME
az acr show --name $ACR_NAME --query loginServer --output table

# Tag container image
acrLoginServer=utbmilbelief.azurecr.io
acrLoginServer=$(az acr show --name $ACR_NAME --query loginServer)
docker tag belief $acrLoginServer/belief:v1

# Push image to ACR (Azure Container Registry)
docker push $acrLoginServer/belief:v1

# List images in Azure Container Registry
az acr repository list --name $ACR_NAME --output table

# Deploy container
CONTAINER_NAME=bmil-belief-app
# this command does not work, maybe some variables are not correctly processed
az container create --resource-group $ACI_PERS_RESOURCE_GROUP --name $CONTAINER_NAME --image $acrLoginServer/belief:v1 --cpu 1 --memory 1 --registry-login-server $acrLoginServer --registry-username utbmilbelief --registry-password pbM5vS0Cw2dGCK2oi/BvleZ2S/WA4a7tzvbGiWf/NZ+ACRBOvl8K --ip-address Public --dns-name-label utbmilbelief --ports 80

az container create --resource-group beliefResourceGroup --name bmil-belief-app --image utbmilbelief.azurecr.io/belief:v1 --cpu 1 --memory 1 --registry-login-server utbmilbelief.azurecr.io --registry-username utbmilbelief --registry-password pbM5vS0Cw2dGCK2oi/BvleZ2S/WA4a7tzvbGiWf/NZ+ACRBOvl8K --ip-address Public --dns-name-label utbmilbelief --ports 80

# Verify deployment progress
az container show --resource-group $ACI_PERS_RESOURCE_GROUP --name $CONTAINER_NAME --query instanceView.state

# View the application and container logs
az container show --resource-group $ACI_PERS_RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn

az container logs --resource-group $ACI_PERS_RESOURCE_GROUP --name $CONTAINER_NAME
```
