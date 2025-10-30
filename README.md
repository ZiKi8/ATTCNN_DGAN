This repository implements ATTCNN_DGAN, a deep learning framework that integrates Convolutional Neural Networks (CNNs), dual-average attention, and Generative Adversarial Networks (GANs) for stock market time-series forecasting. It also includes several baseline models for performance comparison:
•	ATTCNN-DGAN (proposed model)
•	GRU-VAE-WGAN
•	CNN-BiLSTM-AM
•	GRU
•	GRU-BERT-VAE-WGAN
•	ATTCNN_DGAN_No_ATT
•	ATTCNN_DGAN_Conv_ATT
Additionally, the repository contains multiple daily historical stock datasets, including:
•	AAPL
•	META
•	NVDA
•	AMZN
•	MSFT
•	GOOG
Each dataset includes:
•	Price and volume data: Open, High, Low, Close, Volume
•	Technical indicators: SMA, EMA, RSI, ATR, MACD, Bollinger Bands, RSV
•	Frequency-domain features: Fourier components
•	Attention features: Derived through temporal attention modules
This repository contains the proposed model and multiple baselines. The ATTCNN_DGAN follows a modular structure, while baseline models are implemented as single .ipynb Python files. 
├─ models/ 
│ ├─ ATTCNN_DGAN/ # Proposed model 
│ │ ├─ backbone/ # Core components 
│ │ │ ├─ init.py 
│ │ │ ├─ data.py # I/O, scaling, sliding windows 
│ │ │ ├─ generator.py # CNN-based Generator 
│ │ │ ├─ discriminator.py # CNN-based Discriminator 
│ │ │ ├─ attn.py # (Optional) Attention feature extractor 
│ │ │ ├─ train_loop.py # Training process, plotting, results 
│ │ ├─ run_train.py # Entry point for execution

library Requirements Please ensure the following Python libraries are installed before running the code: tensorflow==2.15.0 numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 scipy==1.11.4 matplotlib==3.9.0 yfinance==0.2.41
Optional (for baseline models): torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
Install all dependencies via: pip install -r requirements.txt
Usage Instructions
1.	Navigate to the project directory: cd models/ATTCNN_DGAN
2.	Run the model: python run_train.py

