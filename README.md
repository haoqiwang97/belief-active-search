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

## Database
https://dbdiagram.io/d/belief-study-4-650a07f502bd1c4a5ee2b32c
A copy saved at https://utexas.app.box.com/file/1490506604634

## Commands
Local test
```zsh
pip list --format=freeze > requirements.txt

conda activate stan_env
# conda activate belief_active_search? No, use stan_env
uvicorn main:app --reload

docker build -t belief .
docker run -dp 80:80 belief
# open http://0.0.0.0:80

# use volume
docker volume create belief-db
docker run -dp 80:80 -v belief-db:/app/database belief
```

Azure, deploy container and mount volume, first manually upload database file (command line written below), https://learn.microsoft.com/en-us/azure/container-instances/container-instances-volume-azure-files

```zsh
# sign in BMIL
# az login -u utbmil@gmail.com -p xx, I do not know the password
# interactive login
az login

ACI_PERS_RESOURCE_GROUP=beliefResourceGroup
ACI_PERS_STORAGE_ACCOUNT_NAME=utbeliefstorageaccount
ACI_PERS_LOCATION=eastus
ACI_PERS_SHARE_NAME=beliefshare

# Create resource group
# if not first time, skip this
az group create --name $ACI_PERS_RESOURCE_GROUP --location $ACI_PERS_LOCATION

# Create an Azure Container Registry (ACR)
# previously I used ACR_NAME=bmilbelief, so bmilbelief.azurecr.io is already in use
ACR_NAME=utbmilbelief
# if not first time, skip this
az acr create --resource-group $ACI_PERS_RESOURCE_GROUP --name $ACR_NAME --sku Basic

# log in ACR
az acr login --name $ACR_NAME
az acr show --name $ACR_NAME --query loginServer --output table

# Tag container image
acrLoginServer=utbmilbelief.azurecr.io
# acrLoginServer=$(az acr show --name $ACR_NAME --query loginServer)
docker tag belief $acrLoginServer/belief:v14

# Push image to ACR (Azure Container Registry)
docker push $acrLoginServer/belief:v14

# List images in Azure Container Registry
az acr repository list --name $ACR_NAME --output table

# Create the storage account with the parameters
# if not first time, skip this
az storage account create \
    --resource-group $ACI_PERS_RESOURCE_GROUP \
    --name $ACI_PERS_STORAGE_ACCOUNT_NAME \
    --location $ACI_PERS_LOCATION \
    --sku Standard_LRS

# Create the file share
# if not first time, skip this
az storage share create \
  --name $ACI_PERS_SHARE_NAME \
  --account-name $ACI_PERS_STORAGE_ACCOUNT_NAME

# Storage account key
STORAGE_KEY=$(az storage account keys list --resource-group $ACI_PERS_RESOURCE_GROUP --account-name $ACI_PERS_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)
echo $STORAGE_KEY

# Manually upload database file, depends on whether want to erase the database on cloud
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
# az acr update -n utbmilbelief --admin-enabled true, 20240503
az acr credential show --name $ACR_NAME
REGISTRY_PASSWORD=IlsbNa0DW8+XOGSwYZub9ETUjPOXKY7Asg3caYLw7D+ACRBj+a3B

################################################################################
# type: container instances
# first time
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

# update container, note v1 -> v14
# need 2 cpu, otherwise it cannot run, stan is overkill here, future may use algorithm that needs less cpu
az container create \
    --resource-group $ACI_PERS_RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image utbmilbelief.azurecr.io/belief:v14 \
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

################################################################################

################################################################################
# type: container app
# https://learn.microsoft.com/en-us/azure/container-apps/storage-mounts-azure-files?tabs=bash
STORAGE_MOUNT_NAME="beliefstoragemount"
ENVIRONMENT_NAME="belief-environment"

# first time, todo:
az containerapp env create \
  --name $ENVIRONMENT_NAME \
  --resource-group $ACI_PERS_RESOURCE_GROUP \
  --location $ACI_PERS_LOCATION \
  --query "properties.provisioningState"

# Create the storage link in the environment
az containerapp env storage set \
  --name $ENVIRONMENT_NAME \
  --resource-group $ACI_PERS_RESOURCE_GROUP \
  --access-mode ReadWrite \
  --azure-file-account-name $ACI_PERS_STORAGE_ACCOUNT_NAME \
  --azure-file-account-key $STORAGE_KEY \
  --azure-file-share-name $ACI_PERS_SHARE_NAME \
  --storage-name $STORAGE_MOUNT_NAME \
  --output table

CONTAINER_APP_NAME="belief-app"
# Create the container app
az containerapp create \
    --name $CONTAINER_APP_NAME \
    --resource-group $ACI_PERS_RESOURCE_GROUP \
    --environment $ENVIRONMENT_NAME \
    --image utbmilbelief.azurecr.io/belief:v12 \
    --registry-server $acrLoginServer \
    --registry-username utbmilbelief \
    --registry-password $REGISTRY_PASSWORD \
    --target-port 80 \
    --ingress external \
    --cpu 2 --memory 4.0 \
    --min-replicas 1 \
    --max-replicas 1 \
    --query properties.configuration.ingress.fqdn

# belief-app.mangofield-d1cf3848.eastus.azurecontainerapps.io

# Export the container app's configuration
az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $ACI_PERS_RESOURCE_GROUP \
  --output yaml > app.yaml

# edit app configuration
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $ACI_PERS_RESOURCE_GROUP \
  --yaml app.yaml \
  --output table

# execute commands inside the running container
az containerapp exec --name $CONTAINER_APP_NAME --resource-group $ACI_PERS_RESOURCE_GROUP
################################################################################

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

## Solve "not secure"
2024-05-03
https://learn.microsoft.com/en-us/azure/container-apps/get-started?tabs=bash

```zsh
# example app
az login
az group create --name my-container-apps --location centralus

az containerapp up \
    --name my-container-app \
    --resource-group my-container-apps \    
    --location centralus \
    --environment 'my-container-apps' \
    --image mcr.microsoft.com/k8se/quickstart:latest \
    --target-port 80 \
    --ingress external \
    --query properties.configuration.ingress.fqdn

```


https://learn.microsoft.com/en-us/azure/container-apps/storage-mounts-azure-files?tabs=bash
```zsh
RESOURCE_GROUP="my-container-apps"
ENVIRONMENT_NAME="my-container-apps"
LOCATION="canadacentral"

az containerapp env storage show --name $ENVIRONMENT_NAME \
                                 --resource-group $RESOURCE_GROUP \
                                 --storage-name $STORAGE_ACCOUNT_NAME

az containerapp env storage set \
    --name my-container-app-belief \
    --resource-group $ACI_PERS_RESOURCE_GROUP \
    --storage-name mystorage \
    --storage-type AzureFile \
    --azure-file-account-name <STORAGE_ACCOUNT_NAME> \
    --azure-file-account-key <STORAGE_ACCOUNT_KEY> \
    --azure-file-share-name <STORAGE_SHARE_NAME> \
    --access-mode ReadWrite

az containerapp create \
    --name my-container-app-belief \
    --resource-group my-container-apps \
    --environment 'my-container-apps' \
    --image utbmilbelief.azurecr.io/belief:v10 \
    --target-port 80 \
    --ingress external \
    --registry-server $acrLoginServer \
    --registry-username utbmilbelief \
    --registry-password $REGISTRY_PASSWORD \
    --cpu 2 --memory 4.0 \
    --min-replicas 1 \
    --max-replicas 5 \
    --query properties.configuration.ingress.fqdn

az containerapp logs show -n my-container-app-belief -g my-container-apps


# does not work, memore needs to be 4
az container create \
    --name $CONTAINER_NAME \
    --resource-group $ACI_PERS_RESOURCE_GROUP \

    --image utbmilbelief.azurecr.io/belief:v10 \
    --dns-name-label utbmilbelief \
    --ports 80 \
    --azure-file-volume-account-name $ACI_PERS_STORAGE_ACCOUNT_NAME \
    --azure-file-volume-account-key $STORAGE_KEY \
    --azure-file-volume-share-name $ACI_PERS_SHARE_NAME \
    --azure-file-volume-mount-path /app/database/ \
    --registry-login-server $acrLoginServer --registry-username utbmilbelief --registry-password $REGISTRY_PASSWORD \
    --cpu 2 --memory 3.5
```

# Old
## 2024-05-03

## Before 2024-05-03
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
