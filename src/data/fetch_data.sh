#! /bin/bash

DATA_PATH="./data"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

cd $DATA_PATH

RAW_PATH="./raw"
PRETRAINED_PATH="./pretrained"
PROCESSED_PATH="./processed"
FEATURES_PATH="./features"

if [ ! -d $RAW_PATH ]; then
  mkdir -p $RAW_PATH
fi

cd $RAW_PATH
# gdown --id 1NtS61Uzy-SCaJQ3FksgYFlMU04sKHT3y
# unzip summary_corpus.zip
# rm summary_corpus.zip


cd ".."

if [ ! -d $FEATURES_PATH ]; then
  mkdir -p $FEATURES_PATH
fi

cd $FEATURES_PATH

mkdir -p "./train"
mkdir -p "./test"
mkdir -p "./val"
mkdir -p "./srl"

cd ".."

if [ ! -d $PROCESSED_PATH ]; then
  mkdir -p $PROCESSED_PATH
fi

cd $PROCESSED_PATH

echo "Downloading processed data (POS Tag)"
# gdown --id 1vL8vyfJbaj3i91peTu738jq25N-yhsU3
# gdown --id 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
# unzip word2vec_news.model.wv.vectors.zip
# rm word2vec_news.model.wv.vectors.zip


cd ".."

if [ ! -d $PRETRAINED_PATH ]; then
  mkdir -p $PRETRAINED_PATH
fi

cd $PRETRAINED_PATH

echo "Downloading pretrained models"
gdown --id 1-BDiBCLeBRDh7ue2IZhrU06Lfju3VsYm
unzip wv_240_0.00075.zip
rm wv_240_0.00075.zip
# gdown --id 1vL8vyfJbaj3i91peTu738jq25N-yhsU3
# gdown --id 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
# unzip word2vec_news.model.wv.vectors.zip
# rm word2vec_news.model.wv.vectors.zip

cd ".."
