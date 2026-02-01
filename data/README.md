# Data

Datasets for training and testing models belong in this directory

## Scripts

To summarise the dataset distribution run the following script: `data/scripts/summarise_data.py`

**Manual Dataset Generation**: 

To generate the dataset from raw audio sources, refer to the scripts in `data/scripts/generate_dataset.py`

## Test Data

Film test dataset: `data/test/a-touch-of-zen`

IRMAS test dataset: `data/test/IRMAS`. Download IRMAS test dataset and place in this directory

## Train Dataset 

Download from teams and place in `data/train`

**Dataset summary**

```plaintext
  strings     1063  (12.8%)  (53m 9s)
  brass       888  (10.7%)  (44m 24s)
  percussion  850  (10.3%)  (42m 30s)
  woodwind    710  (8.6%)  (35m 30s)
  sheng       632  (7.6%)  (31m 36s)
  dizi        623  (7.5%)  (31m 9s)
  timpani     609  (7.4%)  (30m 27s)
  erhu        543  (6.6%)  (27m 9s)
  pipa        496  (6.0%)  (24m 48s)
  suona       464  (5.6%)  (23m 12s)
  guzheng     443  (5.3%)  (22m 9s)
  piano       318  (3.8%)  (15m 54s)
  guqin       287  (3.5%)  (14m 21s)
  xiao        239  (2.9%)  (11m 57s)
  yangqin     116  (1.4%)  (5m 48s)

Total clip counts: 8281
  ```

### Suggestions

- Horn (7)	Brass	Shares similar harmonic series and attack.
- Bass (5)	Strings	Assuming this is a Double Bass or Cello.
- Banzi Clapper (11)	Percussion	Distinct, but could go in "Percussion" 
- Electronic (5)	New Class?	If you have 15m of Piano, consider replacing 5m with Synth/Electronic.

## A touch of zen test dataset

Download train dataset from teams under `General/Datasets/train`
Unzip from teams and place in `data/train`

```plaintext
  strings                         45   (12.4%)
  brass                           32   (8.8%)
  sheng                           30   (8.2%)
  woodwind                        30   (8.2%)
  pipa                            29   (8.0%)
  percussion                      28   (7.7%)
  timpani                         26   (7.1%)
  erhu                            23   (6.3%)
  dizi                            12   (3.3%)
  banzi_clapper                   11   (3.0%)
  guqin                           10   (2.7%)
  bell                            8    (2.2%)
  xiao                            8    (2.2%)
  horn                            7    (1.9%)
  qing                            7    (1.9%)
  bass                            5    (1.4%)
  electronic                      5    (1.4%)
  suona                           4    (1.1%)
  triangle                        4    (1.1%)
  voice                           4    (1.1%)
  chanting_scriptures             3    (0.8%)
  gong                            3    (0.8%)
  marimba                         3    (0.8%)
  morin_khuur                     3    (0.8%)
  cymbals                         2    (0.5%)
  drums                           2    (0.5%)
  guzheng                         2    (0.5%)
  operatic_clapper                2    (0.5%)
  operatic_gongs_and_drums        2    (0.5%)
  piano                           2    (0.5%)
  violin                          2    (0.5%)
  bell_toll                       1    (0.3%)
  clapper                         1    (0.3%)
  huiyuans_buddha_light           1    (0.3%)
  operatic_gongs_and_drums_suona  1    (0.3%)
  qing_stone_chime                1    (0.3%)
  snare_drum                      1    (0.3%)
  wind                            1    (0.3%)
  wind_chimes                     1    (0.3%)
  xiao_zen_like                   1    (0.3%)
  yangqin                         1    (0.3%)
```

