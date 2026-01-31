# Data

Datasets for training and testing models belong in this directory

## Train data

Unzip from teams and place in `data/train`

## Scripts

To summarise the dataset distribution run the following script: `data/scripts/summarise_data.py`

To generate the dataset from raw audio sources, refer to the scripts in `data/scripts/generate_dataset.py`

## Test Data

Film test dataset: `data/test/a-touch-of-zen`

IRMAS test dataset: `data/test/IRMAS`. Download IRMAS test dataset and place in this directory

## Train Dataset Summary

```plaintext
Per-label clip counts:

  dizi           623  (~31m 9s)
  erhu           188  (~9m 24s)
  guqin          287  (~14m 21s)
  guzheng        443  (~22m 9s)
  percussion     850  (~42m 30s)
  pia            721  (~36m 3s)
  pipa           141  (~7m 3s)
  sheng          298  (~14m 54s)
  suona          851  (~42m 33s)
  voice          778  (~38m 54s)
  xiao           239  (~11m 57s)
  yangqin        116  (~5m 48s)

Total clip counts: 5535
  ```

## Other classes 

To include in future

```plaintext
Timpani
Cymbals
Horn
Electronic
Singing
```