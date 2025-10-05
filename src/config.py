import os

DB_URL = os.environ.get("DF_DB_URL", "sqlite:///data/retail.db")
FORECAST_HORIZON = int(os.environ.get("DF_HORIZON", "90"))
VAL_DAYS = int(os.environ.get("DF_VAL_DAYS", "90"))
RANDOM_SEED = int(os.environ.get("DF_SEED", "42"))
