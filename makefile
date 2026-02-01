VENV_PY := $(firstword $(wildcard .venv/bin/python) $(wildcard .venv/Scripts/python.exe))
PY ?= $(if $(VENV_PY),$(VENV_PY),python)

PY_SRC := PYTHONPATH=src $(PY)
PROCESSED_ROOT := data/processed
CONFIG_FILE := src/configs/audio_params.yaml
LABELS_CONFIG := src/configs/labels.yaml

# Mix train mels and gennerate spectrogram
NUM_MIXES ?= 20000
MIN_SOURCES ?= 2
MAX_SOURCES ?= 2
SNR_DB_MIN ?= -5
SNR_DB_MAX ?= 10
MIX_SEED ?= 1337
NUM_MIXES ?= 500

MIXED_CACHE_ROOT := $(PROCESSED_ROOT)/log_mels_mixed
MIXED_MANIFEST := $(PROCESSED_ROOT)/train_mels_mixed.csv
TRAIN_DIR := data/train

generate_mixed_train_mels:
	$(PY_SRC) src/scripts/generate_mixed_train_mels.py \
		--config $(CONFIG_FILE) \
		--labels_file $(LABELS_CONFIG) \
		--train_dir $(TRAIN_DIR) \
		--out_cache_root $(MIXED_CACHE_ROOT) \
		--out_manifest $(MIXED_MANIFEST) \
		--num_mixes $(NUM_MIXES) \
		--seed $(MIX_SEED) \
		--save_wavs \
		--wav_out_dir $(PROCESSED_ROOT)/debug/mixed_wavs \
		--max_wavs 50

generate_train_mels:
	$(PY_SRC) src/scripts/generate_log_mels.py \
		--config $(CONFIG_FILE) \
		--labels_file $(LABELS_CONFIG)

TEST_DIR_AZ := data/test/a-touch-of-zen
TEST_MANIFEST_AZ := $(TEST_DIR_AZ).csv
TEST_DIR_IRMAS := data/test/IRMAS/IRMAS-TestingData-Part1
TEST_MANIFEST_IRMAS := $(TEST_DIR_IRMAS).csv

generate_features: generate_train_mels generate_mixed_train_mels

test_manifest:
	@echo "Creating test manifest..."
	$(PY_SRC) src/scripts/generate_test_manifest.py \
		--test_dir $(TEST_DIR) \
		--out_csv $(OUT_CSV)

test_manifest_az:
	@$(MAKE) test_manifest TEST_DIR=$(TEST_DIR_AZ) OUT_CSV=$(TEST_MANIFEST_AZ)

test_manifest_irmas:
	@$(MAKE) test_manifest TEST_DIR=$(TEST_DIR_IRMAS) OUT_CSV=$(TEST_MANIFEST_IRMAS)

clean:
	rm -rf $(PROCESSED_ROOT)
