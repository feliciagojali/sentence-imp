#! /bin/bash

MODEL_PATH="./models"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

DATA_PATH="./data"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

cd $DATA_PATH

RAW_PATH="./raw"
PRETRAINED_PATH="./pretrained"
RESULTS_PATH="./results"
FEATURES_PATH="./features"

if [ ! -d $RAW_PATH ]; then
  mkdir -p $RAW_PATH
fi

cd $RAW_PATH
gdown 1trA-Z9jYexgfw5OurEvIvx_uxIdLQmer
unzip summary_corpus.csv

cd ".."

if [ ! -d $FEATURES_PATH ]; then
  mkdir -p $FEATURES_PATH
fi

cd ".."

if [ ! -d $RESULTS_PATH ]; then
  mkdir -p $RESULTS_PATH
fi


if [ ! -d $PRETRAINED_PATH ]; then
  mkdir -p $PRETRAINED_PATH
fi

cd $PRETRAINED_PATH

echo "Downloading pretrained models"
gdown --id 1553_9shAUrQpFAB0vqXbQrpHqJvWYgp4
unzip word2vec-input_sent.txt-s300-c5-w5-e10-SG.model.trainables.syn1neg.zip
rm word2vec-input_sent.txt-s300-c5-w5-e10-SG.model.trainables.syn1neg.zip
gdown --id 1vL8vyfJbaj3i91peTu738jq25N-yhsU3
gdown --id 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
unzip word2vec_news.model.wv.vectors.zip
rm word2vec_news.model.wv.vectors.zip
cd ".."
