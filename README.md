# ML_based_analysis_of_sound

## Machine Learning-Based Analysis of Music and Sound in Martial Arts Films

[Project tasks](https://github.com/users/hughmancoder/projects/4)

## Setup

Install prequisites on your machine
`git, python3, pip, make`

```bash
# Create virtual environment
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate   

# On Windows (cmd.exe)
.venv\Scripts\activate.bat

# On Windows (PowerShell)
. .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Activate environment (venv) on every terminal

## Run the project

refer to the make file for command lines

```bash
make help
```

## Quickstart (recommended flow)

1) Prepare datasets (see `data/README.md`).
2) Generate mel features.
3) (Optional) Generate mixed mel features.
4) (Optional) Generate CQT or Mel+CQT features.
5) Train (run `.py` or `.ipynb`).
6) Test (run `.py` or `.ipynb`).

### Common preprocessing targets

```bash
# Chinese instruments (mel)
make generate_train_mels

# Mixed mel (multilabel mixes)
make generate_mixed_train_mels

# Chinese instruments (CQT, aligned to existing mel manifest)
make generate_chinese_train_cqt

# IRMAS train/test (mel)
make generate_irmas_train_mels
make test_manifest_irmas

# IRMAS train/test (CQT)
make generate_irmas_train_cqt
make generate_irmas_test_cqt

# Generate all preprocessing (no training)
make all
```

### Training scripts (CLI)

```bash
# Chinese mel-only
python src/train/train_chinese_mel.py

# Chinese mel+cqt
python src/train/train_chinese_mel_cqt.py

# IRMAS mel-only
python src/train/train_irmas_mel.py

# IRMAS mel+cqt
python src/train/train_irmas_mel_cqt.py
```

### Training notebooks (UI click-run)

Open and run in VSCode/Jupyter:

- `src/train/train_chinese_mel.ipynb`
- `src/train/train_chinese_mel_cqt.ipynb`
- `src/train/train_irmas_mel.ipynb`
- `src/train/train_irmas_mel_cqt.ipynb`

### Test scripts (CLI)

```bash
# Chinese mel-only
python src/test/test.py

# Chinese mel+cqt
python src/test/test_chinese_mel_cqt.py

# IRMAS mel+cqt
python src/test/test_irmas_mel_cqt.py
```

### Test notebooks (UI click-run)

- `src/test/test_chinese_mel.ipynb` (Chinese mel-only)
- `src/test/test_irmas_mel.ipynb` (IRMAS mel-only)
- `src/test/test_chinese_mel_cqt.ipynb`
- `src/test/test_irmas_mel_cqt.ipynb`

## FAQ: CPU满载、GPU占用低怎么办？

**结论**：这通常不是“神经网络架构太弱”，而是**数据准备/读取在CPU上成为瓶颈**，GPU在等数据。

常见原因：
- Mel/CQT 生成是 CPU 计算（librosa/np），本身不走 GPU。
- DataLoader 读取 `.npy` + 拼接/归一化在 CPU。
- Windows 下 `num_workers>0` 有时反而变慢（进程启动/拷贝开销大）。
- batch 太小，GPU 吃不饱。

你可以尝试的提速方向（训练阶段）：

1) **增大 batch size**
   - GPU 利用率提升明显，但显存会增加。
2) **确保已经预处理**（mel/cqt 都生成好）
   - 训练时不要在线计算特征。
3) **DataLoader 优化**
   - 在 `utils_mel_cqt.py` / `utils.py` 里使用 `pin_memory=True`、`persistent_workers=True`（Windows 下要谨慎评估）。
4) **混合精度**
   - 如果模型支持 AMP，训练脚本可以启用 `torch.cuda.amp.autocast`。
5) **更大的模型或更长序列**
   - 但这通常不是首选，先把 I/O 和 batch 弄好。

判断瓶颈的方法：
- GPU 利用率长期很低（<30%）且 CPU 满载 → I/O 或 CPU 预处理是瓶颈。
- GPU 利用率高且显存接近满 → 模型/算力是瓶颈。

如果你愿意，我可以帮你把 DataLoader 的 pin_memory/persistent_workers 方案加进去，并做 Windows 兼容配置。

### Launch the Gradio interface

After installing dependencies, start the inference GUI with:

```bash
make run_gradio_gui
```

This launches the two-tab Gradio app (Model + Info) using the fine-tuned weights at `saved_weights/chinese_single_class/train_1/best_val_acc.pt` by default. Upload or record a ~3 second clip, inspect the generated mel spectrogram, and review the predicted class
probabilities in the browser.

## Datasets

Refer to data README.md [here](data/README.md) for details on datasets
