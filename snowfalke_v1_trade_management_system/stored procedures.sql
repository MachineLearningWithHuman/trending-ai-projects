-- Create or replace the stored procedure
CREATE OR REPLACE PROCEDURE get_historical_trades(offset_minutes INT)
RETURNS TABLE(
    ORDER_ID INT,              -- Ensure this matches the actual data type in the table
    TRADE_DATE TIMESTAMP, 
    SYMBOL STRING, 
    TRADE_TYPE STRING, 
    QUANTITY INT,
    PRICE FLOAT, 
    STATUS STRING,
    EXECUTION_TIME TIMESTAMP,
    MARKET_PRICE FLOAT, 
    SLIPPAGE FLOAT, 
    TRADE_FLAG STRING
)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'get_historical_trades'
AS
$$
import snowflake.snowpark as snowpark

def get_historical_trades(session, offset_minutes: int):
    """
    Fetches historical trades using Time Travel.
    The offset_minutes parameter specifies how many minutes back in time to look.
    """
    # Query the table using Time Travel and filter by the calculated time range
    df = session.sql(f"""
        SELECT * 
        FROM trade_orders
        AT(OFFSET => -60 * {offset_minutes});  -- Offset in seconds
    """)
    return df
$$;

select current_timestamp();
SELECT * FROM trade_orders;
CALL get_historical_trades(10);

CALL get_trade_analysis();


-- Create or replace the stored procedure
CREATE OR REPLACE PROCEDURE get_trade_analysis()
RETURNS TABLE(
    trade_date TIMESTAMP, 
    symbol STRING, 
    trade_type STRING, 
    price FLOAT, 
    market_price FLOAT, 
    slippage FLOAT, 
    execution_time TIMESTAMP, 
    trade_flag STRING, 
    execution_delay INTEGER
)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'get_trade_analysis'
AS
$$
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col

def get_trade_analysis(session):
    """
    Analyzes execution delays, slippage, and trade anomalies.
    """
    df = session.table("trade_orders") \
        .select("trade_date", "symbol", "trade_type", "price", "market_price", "slippage", "execution_time", "trade_flag") \
        .with_column("execution_delay", snowpark.functions.datediff("second", col("trade_date"), col("execution_time"))) \
        .filter(col("trade_flag") != 'NORMAL')  # Get only flagged trades

    return df
$$;