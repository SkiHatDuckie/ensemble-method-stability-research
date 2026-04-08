#! bin/bash
echo Downloading Telco customer churn datset to "$PWD"/data/
DATA_PATH=data/
FILE="$DATA_PATH"/telco-customer-churn.zip
curl -L -o "$FILE"\
  https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn\
  --create-dirs
unzip "$FILE" -d "$DATA_PATH"