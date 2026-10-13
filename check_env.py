import torch
import pythainlp
import sklearn
import pandas as pd

print('=== System Check ===')
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
print('PyThaiNLP Version:', pythainlp.__version__)
print('Scikit-Learn Version:', sklearn.__version__)
print('Pandas Version:', pd.__version__)
print('All systems ready!')
