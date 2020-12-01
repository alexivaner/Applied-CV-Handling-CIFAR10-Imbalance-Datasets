
Install Dependencies:
	conda install seaborn, matplotlib, NumPy

HOW TO RUN: 
Default (Without resampling and weight rebalancing)
	python 0860812.py

Turn on Resampling only
	python 0860812.py --resampling_balance True

Turn on loss reweighting only
	python 0860812.py -- reweight_balance True	
