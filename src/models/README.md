# Model Comparison Log

## TO TRY

### Models

CNN 
CNN14
CRNN

### Features

log-mel
log-mel + wavegram
log-mel + CQT

## CNN

### v0

Notes: trained on mels
Weights: `src/models/saved_weights/CNN_v0`
Features: Log-Mel
Classification threshold probability: 0.75
Total Samples: 85
Predicted 'None' (all-zero): 24  (28.24%)
Accuracy: 73.41%
Subset accuracy (exact match): 2.35%
Micro-Average F1: 0.0174
Macro-Average F1: 0.0183

### v1

Notes: trained on trained on pure and mixed waveform mels
