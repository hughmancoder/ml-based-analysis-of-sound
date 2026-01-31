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

### Launch the Gradio interface

After installing dependencies, start the inference GUI with:

```bash
make run_gradio_gui
```

This launches the two-tab Gradio app (Model + Info) using the fine-tuned weights at `saved_weights/chinese_single_class/train_1/best_val_acc.pt` by default. Upload or record a ~3 second clip, inspect the generated mel spectrogram, and review the predicted class
probabilities in the browser.

## Datasets

Refer to data README.md [here](data/README.md) for details on datasets
