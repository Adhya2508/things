import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("dataset.csv")

engine = create_engine(
    "mysql+pymysql://user:pass123@127.0.0.1:3306/bank_db"
)

df.to_sql(
    "cc_general",
    con=engine,
    if_exists="replace",
    index=False
)

print("Imported successfully!")
