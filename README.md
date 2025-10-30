This repository implements ATTCNN_DGAN, a deep learning framework that integrates Convolutional Neural Networks (CNNs), dual-average attention, and Generative Adversarial Networks (GANs) for stock market time-series forecasting. It also includes several baseline models for performance comparison:<br>
•	ATTCNN-DGAN (proposed model)<br>
•	GRU-VAE-WGAN<br>
•	CNN-BiLSTM-AM<br>
•	GRU<br>
•	GRU-BERT-VAE-WGAN<br>
•	ATTCNN_DGAN_No_ATT<br>
•	ATTCNN_DGAN_Conv_ATT<br>
Additionally, the repository contains multiple daily historical stock datasets, including:<br>
•	AAPL<br>
•	META<br>
•	NVDA<br>
•	AMZN<br>
•	MSFT<br>
•	GOOG<br><br>
Each dataset includes:<br>
•	Price and volume data: Open, High, Low, Close, Volume<br>
•	Technical indicators: SMA, EMA, RSI, ATR, MACD, Bollinger Bands, RSV<br>
•	Frequency-domain features: Fourier components<br>
•	Attention features: Derived through temporal attention modules<br>
This repository contains the proposed model and multiple baselines. The ATTCNN_DGAN follows a modular structure, while baseline models are implemented as single .ipynb Python files. <br>
├─ models/ <br>
│ ├─ ATTCNN_DGAN/ # Proposed model <br>
│ │ ├─ backbone/ # Core components <br>
│ │ │ ├─ init.py <br>
│ │ │ ├─ data.py # I/O, scaling, sliding windows <br>
│ │ │ ├─ generator.py # CNN-based Generator <br>
│ │ │ ├─ discriminator.py # CNN-based Discriminator <br>
│ │ │ ├─ attn.py # (Optional) Attention feature extractor <br>
│ │ │ ├─ train_loop.py # Training process, plotting, results <br>
│ │ ├─ run_train.py # Entry point for execution <br>

library Requirements Please ensure the following Python libraries are installed before running the code: <br>
tensorflow==2.15.0 <br>
numpy==1.26.4 <br>
pandas==2.2.2 <br>
scikit-learn==1.4.2 <br>
scipy==1.11.4 <br>
matplotlib==3.9.0 <br>
yfinance==0.2.41
Optional (for baseline models): <br>
torch==2.2.2 <br>
torchvision==0.17.2 <br>
torchaudio==2.2.2<br>
Install all dependencies via: pip install -r requirements.txt <br>
Usage Instructions<br>
1.	Navigate to the project directory: cd models/ATTCNN_DGAN<br>
2.	Run the model: python run_train.py<br>

