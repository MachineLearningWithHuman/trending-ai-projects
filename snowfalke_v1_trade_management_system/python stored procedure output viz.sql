# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col,datediff

def main(session: snowpark.Session): 
    # Your code goes here, inside the "main" handler.
    #tableName = 'information_schema.packages'
    #dataframe = session.table(tableName).filter(col("language") == 'python')
    df = session.sql(f"""
        SELECT * FROM trade_orders AT(OFFSET => -60*5)
    """)

    df = session.table("trade_orders") \
        .select("trade_date", "symbol", "trade_type", "price", "market_price", "slippage", "execution_time", "trade_flag") \
        .with_column("execution_delay", datediff("second", col("trade_date"), col("execution_time"))) \
        .filter(col("trade_flag") != 'NORMAL')  # Get only flagged trades

    #return df

    # Print a sample of the dataframe to standard output.
    df.show()

    # Return value will appear in the Results tab.
    return df