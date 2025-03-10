USE DATABASE YOUTUBE;

CREATE OR REPLACE SCHEMA EXAMPLE;


CREATE TABLE trade_orders (
    order_id INT AUTOINCREMENT,
    trade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol STRING,
    trade_type STRING, -- "BUY" or "SELL"
    quantity INT,
    price FLOAT,
    status STRING DEFAULT 'PENDING' -- "PENDING", "EXECUTED"
);

INSERT INTO trade_orders (symbol, trade_type, quantity, price, status) VALUES
    ('AAPL', 'BUY', 10, 175.50, 'PENDING'),
    ('TSLA', 'SELL', 5, 890.30, 'PENDING'),
    ('GOOGL', 'BUY', 15, 2750.80, 'PENDING'),
    ('MSFT', 'SELL', 8, 310.20, 'PENDING'),
    ('NFLX', 'BUY', 12, 605.75, 'PENDING'),
    ('AMZN', 'SELL', 7, 3400.90, 'PENDING'),
    ('META', 'BUY', 20, 320.45, 'PENDING'),
    ('NVDA', 'SELL', 6, 245.10, 'PENDING'),
    ('TSLA', 'BUY', 9, 880.90, 'PENDING'),
    ('AAPL', 'SELL', 4, 176.80, 'PENDING');


    INSERT INTO trade_orders (symbol, trade_type, quantity, price, status) VALUES
    ('AAPL', 'BUY', 100, 175.50, 'PENDING');


SELECT * FROM trade_orders AT (TIMESTAMP => '2025-03-10 12:00:00');
SELECT * FROM trade_orders AT (OFFSET => -60*1);
SELECT * FROM trade_orders BEFORE (STATEMENT => 'QUERY_ID');
-- Time Travel: Fetch trades as they were 10 minutes ago
SELECT * FROM trade_orders AT(TIMESTAMP => DATEADD(MINUTE, -10, CURRENT_TIMESTAMP));


DROP TABLE TRADE_ORDERS;

UNDROP TABLE TRADE_ORDERS;

CREATE or replace TABLE TRADE_ORDERS_clone CLONE TRADE_ORDERS AT (OFFSET => -60*1);

ALTER TABLE TRADE_ORDERS SET DATA_RETENTION_TIME_IN_DAYS = 7;

-- STREAM 

CREATE STREAM TRADE_ORDERS_stream ON TABLE TRADE_ORDERS;

SELECT * FROM TRADE_ORDERS_stream;

INSERT INTO trade_orders (symbol, trade_type, quantity, price, status) VALUES
    ('AAPL', 'BUY', 100, 175, 'PENDING');


SELECT * FROM TRADE_ORDERS_stream;


CREATE OR REPLACE TASK process_my_stream
WAREHOUSE = EXAMPLE_WH
SCHEDULE = '1 MINUTE'
AS 
INSERT INTO TRADE_ORDERS_clone
SELECT * FROM TRADE_ORDERS_stream;


ALTER TASK process_my_stream RESUME;


SHOW TASKS;

SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY());


//ALTER ACCOUNT SET ENABLE_NATIVE_PYTHON_UDFS = TRUE;







