#!/bin/bash

MODEL_NAME=fire_detection
MODEL_LOCATION="gs://fire_detection_anurag/models/accuracy/chkpts"  # will change when we call script
# MODEL_LOCATION="export/explainable_fire_model/saved_model.pb"
VERSION_NAME=ig
REGION='us-central1'  # You need GPU quota in this region
EXPLAIN="--explanation-method integrated-gradients --num-integral-steps 25"

BUCKET='fire_detection_staging'  # for staging


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -v|--version) VERSION_NAME="$2"; shift ;;
        -m|--model_location) MODEL_LOCATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$VERSION_NAME" -eq "xrai" ]]; then
   EXPLAIN="--explanation-method xrai --num-integral-steps 25"
fi

echo "Deploying $MODEL_NAME:$VERSION_NAME with $EXPLAIN from $MODEL_LOCATION"

if [[ $(gcloud ai-platform models list --region=$REGION --format='value(name)' | grep $MODEL_NAME) ]]; then
    echo "The model named $MODEL_NAME already exists."
else
    # create model
    echo "Creating $MODEL_NAME model now."
    gcloud ai-platform models create --region=$REGION --enable-logging $MODEL_NAME
fi

if [[ $(gcloud ai-platform versions list --region=$REGION --model $MODEL_NAME --format='value(name)' | grep $VERSION_NAME) ]]; then
    echo "Deleting the already existing model $MODEL_NAME:$VERSION_NAME ... "
    gcloud ai-platform versions delete --quiet --region=$REGION --model=$MODEL_NAME $VERSION_NAME
    echo "Please run this script again if you don't see a Creating message ... "
    sleep 2
fi

# create model
echo "Creating $MODEL_NAME:$VERSION_NAME $EXPLAIN"
gcloud beta ai-platform versions create --model=$MODEL_NAME $VERSION_NAME --async \
       --region=$REGION  --machine-type n1-standard-4 \
       --framework=tensorflow --python-version=3.7 --runtime-version=2.6 \
       --origin=$MODEL_LOCATION --staging-bucket=gs://$BUCKET $EXPLAIN
