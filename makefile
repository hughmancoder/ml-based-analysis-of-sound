VENV_PY := $(firstword $(wildcard .venv/bin/python) $(wildcard .venv/Scripts/python.exe))
PY ?= $(if $(VENV_PY),$(VENV_PY),python)

PY_SRC := PYTHONPATH=src $(PY)
PROCESSED_ROOT := data/processed
CONFIG_FILE := src/configs/audio_params.yaml
LABELS_CONFIG := src/configs/labels.yaml
NUM_WORKERS ?= 19
IRMAS_CONFIG := src/configs/audio_params_irmas.yaml
IRMAS_LABELS := src/configs/labels_irmas.yaml
IRMAS_TRAIN_DIR := data/audio/IRMAS/IRMAS-TrainingData/IRMAS-TrainingData
IRMAS_TRAIN_MELS_MANIFEST := $(PROCESSED_ROOT)/irmas_train_mels.csv
CQT_CACHE_ROOT := $(PROCESSED_ROOT)/log_cqt
IRMAS_CQT_CACHE_ROOT := $(PROCESSED_ROOT)/irmas_cqt
IRMAS_TEST_CQT_CACHE_ROOT := $(PROCESSED_ROOT)/irmas_cqt_test
IRMAS_FMAX := 20000

# Mix train mels and gennerate spectrogram
NUM_MIXES ?= 20000
MIN_SOURCES ?= 2
MAX_SOURCES ?= 2
SNR_DB_MIN ?= -3
SNR_DB_MAX ?= 6
NUM_MIXES ?= 12000 # start with ~150% of dataset size

MIXED_CACHE_ROOT := $(PROCESSED_ROOT)/log_mels_mixed
MIXED_CQT_CACHE_ROOT := $(PROCESSED_ROOT)/log_cqt_mixed
MIXED_MANIFEST := $(PROCESSED_ROOT)/train_mels_mixed.csv
TRAIN_DIR := data/train

# NOTE: premixing will be replaced with mixing at train time to save storage
generate_mixed_train_mels:
	$(PY_SRC) src/scripts/generate_mixed_train_mels.py \
		--config $(CONFIG_FILE) \
		--labels_file $(LABELS_CONFIG) \
		--train_dir $(TRAIN_DIR) \
		--out_cache_root $(MIXED_CACHE_ROOT) \
		--out_manifest $(MIXED_MANIFEST) \
		--num_mixes $(NUM_MIXES) \
		--num_workers $(NUM_WORKERS) \
		--save_wavs \
		--wav_out_dir $(PROCESSED_ROOT)/debug/mixed_wavs \
		--max_wavs 50

generate_mixed_train_mel_cqt:
	$(PY_SRC) src/scripts/generate_mixed_train_mel_cqt.py \
		--config $(CONFIG_FILE) \
		--labels_file $(LABELS_CONFIG) \
		--train_dir $(TRAIN_DIR) \
		--out_cache_root $(MIXED_CACHE_ROOT) \
		--out_cqt_root $(MIXED_CQT_CACHE_ROOT) \
		--out_manifest $(MIXED_MANIFEST) \
		--num_mixes $(NUM_MIXES) \
		--num_workers $(NUM_WORKERS) \
		--save_wavs \
		--wav_out_dir $(PROCESSED_ROOT)/debug/mixed_wavs \
		--max_wavs 50

generate_train_mels:
	$(PY_SRC) src/scripts/generate_log_mels.py \
		--config $(CONFIG_FILE) \
		--labels_file $(LABELS_CONFIG) \
		--num_workers $(NUM_WORKERS)

generate_irmas_train_mels:
	$(PY_SRC) src/scripts/generate_log_mels.py \
		--config $(IRMAS_CONFIG) \
		--labels_file $(IRMAS_LABELS) \
		--train_dir $(IRMAS_TRAIN_DIR) \
		--num_workers $(NUM_WORKERS)

generate_chinese_train_cqt:
	$(PY_SRC) src/scripts/generate_chinese_train_cqt.py \
		--config $(CONFIG_FILE) \
		--labels_file $(LABELS_CONFIG) \
		--train_dir $(TRAIN_DIR) \
		--cqt_cache_root $(CQT_CACHE_ROOT) \
		--num_workers $(NUM_WORKERS)

generate_irmas_train_cqt:
	$(PY_SRC) src/scripts/generate_irmas_train_cqt.py \
		--irmas_train_dir $(IRMAS_TRAIN_DIR) \
		--cache_root $(IRMAS_CQT_CACHE_ROOT) \
		--mel_manifest_out $(IRMAS_TRAIN_MELS_MANIFEST) \
		--fmax $(IRMAS_FMAX) \
		--num_workers $(NUM_WORKERS)

convert_mp3_wav:
	$(PY_SRC) src/scripts/convert_mp3_to_wav.py \
		--root data/audio/chinese_instruments \
		--sr 44100 \
		--channels 2

TEST_DIR_AZ := data/test/a-touch-of-zen
TEST_MANIFEST_AZ := $(TEST_DIR_AZ).csv
TEST_DIR_IRMAS := data/audio/IRMAS/IRMAS-TestingData-Part1/IRMAS-TestingData-Part1/Part1
TEST_MANIFEST_IRMAS := data/test/IRMAS/IRMAS-TestingData-Part1.csv

generate_features: generate_train_mels generate_mixed_train_mels

# One-shot pipeline (ordered)
all: \
	convert_mp3_wav \
	generate_train_mels \
	generate_chinese_train_cqt \
	generate_mixed_train_mel_cqt \
	generate_irmas_train_mels \
	generate_irmas_train_cqt \
	test_manifest_az \
	test_manifest_irmas \
	generate_irmas_test_cqt

test_manifest:
	@echo "Creating test manifest..."
	$(PY_SRC) src/scripts/generate_test_manifest.py \
		--test_dir $(TEST_DIR) \
		--out_csv $(OUT_CSV)

test_manifest_az:
	@$(MAKE) test_manifest TEST_DIR=$(TEST_DIR_AZ) OUT_CSV=$(TEST_MANIFEST_AZ)

test_manifest_irmas:
	@$(MAKE) test_manifest TEST_DIR=$(TEST_DIR_IRMAS) OUT_CSV=$(TEST_MANIFEST_IRMAS)

generate_irmas_test_cqt:
	$(PY_SRC) src/scripts/generate_irmas_test_cqt.py \
		--input_dir $(TEST_DIR_IRMAS) \
		--cache_root $(IRMAS_TEST_CQT_CACHE_ROOT) \
		--manifest_out $(TEST_MANIFEST_IRMAS) \
		--project_root . \
		--dataset_name IRMAS \
		--fmax $(IRMAS_FMAX) \
		--num_workers $(NUM_WORKERS)

clean:
	rm -rf $(PROCESSED_ROOT)
