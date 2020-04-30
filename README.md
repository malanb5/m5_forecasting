# M5 Forecasting - Accuracy
https://www.kaggle.com/c/m5-forecasting-accuracy

Predict the sales of 3049 products in 10 Walmart stores, using historical data.


### Prerequisites
anaconda

### Quickstart

```
conda create --name <env> --file requirements.txt
conda activate <env>

# execute this for the LGBM algorithm
python YJ/WalRunner.py --algorithm lgbm

# execute this for Facebook's Prophet forecasting
python YJ/WalRunner.py --algorithm prophet
```