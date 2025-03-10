CREATE OR REPLACE DATABASE YOUTUBE_CLONE CLONE YOUTUBE AT(OFFSET => -60*40);

CREATE OR REPLACE DATABASE YOUTUBE;

CREATE OR REPLACE SCHEMA DAY_1;

CREATE TABLE trade_orders (
    order_id INT AUTOINCREMENT,
    trade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol STRING,
    trade_type STRING, -- "BUY" or "SELL"
    quantity INT,
    price FLOAT,
    status STRING DEFAULT 'PENDING'
)
DATA_RETENTION_TIME_IN_DAYS = 7;

CREATE STREAM trade_orders_stream ON TABLE trade_orders;

CREATE OR REPLACE TASK process_trades_task
WAREHOUSE = EXAMPLE_WH
SCHEDULE = '1 MINUTE'
WHEN SYSTEM$STREAM_HAS_DATA('trade_orders_stream')
AS
BEGIN
    UPDATE trade_orders
    SET status = 'EXECUTED'
    WHERE order_id IN (SELECT order_id FROM trade_orders_stream);

    -- Simulate Market Price for comparison
    UPDATE trade_orders
    SET market_price = price  * UNIFORM(0.92, 0.99, RANDOM()),  -- Random  fluctuation
        execution_time = CURRENT_TIMESTAMP,
        status = 'EXECUTED',
        slippage = ABS(price - market_price)
    WHERE order_id IN (SELECT order_id FROM trade_orders_stream);
    
    -- Flag trades with high slippage
    UPDATE trade_orders
    SET trade_flag = 'HIGH_SLIPPAGE'
    WHERE slippage > 5; -- Flag if slippage exceeds $5
    
END;


ALTER TASK process_trades_task RESUME;

EXECUTE TASK process_trades_task;-- Start the Task

--GRANT EXECUTE TASK ON ACCOUNT TO ROLE SYSADMIN;


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

select * from trade_orders;

SELECT COUNT(*) FROM TRADE_ORDERS;

SELECT * FROM trade_orders_stream;

INSERT INTO trade_orders (symbol, trade_type, quantity, price, status) VALUES
    ('AAPL', 'BUY', 10, 175.50, 'PENDING'),
    ('TSLA', 'SELL', 5, 890.30, 'PENDING'),
    ('GOOGL', 'BUY', 15, 2750.80, 'PENDING'),
    ('MSFT', 'SELL', 8, 310.20, 'PENDING');

INSERT INTO trade_orders (symbol, trade_type, quantity, price, status, trade_date) VALUES
    ('AAPL', 'BUY', 10, 175.50, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'SELL', 5, 890.30, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'BUY', 15, 2750.80, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'SELL', 8, 310.20, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'BUY', 12, 605.75, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'SELL', 7, 3400.90, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'BUY', 20, 320.45, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'SELL', 6, 245.10, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'BUY', 9, 880.90, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'SELL', 4, 176.80, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'BUY', 25, 178.50, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'SELL', 30, 875.60, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'BUY', 40, 2800.30, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'SELL', 35, 305.50, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'BUY', 20, 610.20, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'SELL', 15, 3380.75, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'BUY', 18, 325.60, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'SELL', 22, 255.30, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'BUY', 17, 890.90, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'SELL', 28, 172.30, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'BUY', 50, 860.40, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'SELL', 45, 300.20, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'BUY', 60, 2900.10, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'SELL', 32, 620.80, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'BUY', 27, 3320.50, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'SELL', 30, 310.70, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'BUY', 14, 260.90, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'SELL', 55, 870.20, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'BUY', 19, 180.40, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'SELL', 33, 2780.80, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'BUY', 29, 315.10, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'SELL', 24, 590.20, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'BUY', 38, 3350.30, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'SELL', 42, 295.50, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'BUY', 25, 270.40, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'SELL', 31, 855.90, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'BUY', 45, 165.20, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'SELL', 48, 2760.40, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'BUY', 52, 320.70, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'SELL', 39, 600.90, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'BUY', 21, 3405.80, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'SELL', 50, 300.60, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'BUY', 33, 280.70, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'SELL', 47, 840.20, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'BUY', 60, 170.30, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'SELL', 55, 2725.90, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'BUY', 37, 330.50, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'SELL', 28, 615.40, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'BUY', 48, 3290.10, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'SELL', 36, 285.20, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'BUY', 41, 290.30, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'SELL', 53, 820.80, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'BUY', 70, 175.90, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'SELL', 62, 2710.50, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'BUY', 54, 340.80, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'SELL', 43, 625.60, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'BUY', 50, 3285.70, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'SELL', 58, 280.30, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'BUY', 39, 295.10, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'SELL', 65, 800.40, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'BUY', 75, 180.10, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'SELL', 70, 2700.30, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'BUY', 65, 350.90, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'SELL', 50, 635.50, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'BUY', 55, 3275.40, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'SELL', 65, 275.20, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'BUY', 48, 300.90, 'PENDING', CURRENT_TIMESTAMP),
    ('TSLA', 'SELL', 72, 780.30, 'PENDING', CURRENT_TIMESTAMP),
    ('AAPL', 'BUY', 80, 185.60, 'PENDING', CURRENT_TIMESTAMP),
    ('GOOGL', 'SELL', 75, 2690.40, 'PENDING', CURRENT_TIMESTAMP),
    ('MSFT', 'BUY', 72, 360.20, 'PENDING', CURRENT_TIMESTAMP),
    ('NFLX', 'SELL', 55, 645.80, 'PENDING', CURRENT_TIMESTAMP),
    ('AMZN', 'BUY', 60, 3265.10, 'PENDING', CURRENT_TIMESTAMP),
    ('META', 'SELL', 70, 270.40, 'PENDING', CURRENT_TIMESTAMP),
    ('NVDA', 'BUY', 55, 305.20, 'PENDING', CURRENT_TIMESTAMP);



ALTER TABLE trade_orders ADD COLUMN execution_time TIMESTAMP;
ALTER TABLE trade_orders ADD COLUMN market_price FLOAT;
ALTER TABLE trade_orders ADD COLUMN slippage FLOAT;
ALTER TABLE trade_orders ADD COLUMN trade_flag STRING DEFAULT 'NORMAL';


INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(86, 'AAPL', 'BUY', 145.25, 100, '2025-03-10 09:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(87, 'AAPL', 'SELL', 146.00, 50, '2025-03-10 09:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(88, 'GOOG', 'BUY', 2800.75, 20, '2025-03-10 09:45:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(89, 'GOOG', 'SELL', 2825.50, 10, '2025-03-10 10:00:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(90, 'AMZN', 'BUY', 3450.80, 15, '2025-03-10 10:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(91, 'AMZN', 'SELL', 3465.25, 5, '2025-03-10 10:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(92, 'TSLA', 'BUY', 650.50, 10, '2025-03-10 10:45:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(93, 'TSLA', 'SELL', 652.25, 10, '2025-03-10 11:00:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(94, 'NFLX', 'BUY', 680.10, 25, '2025-03-10 11:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(95, 'NFLX', 'SELL', 682.75, 10, '2025-03-10 11:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(96, 'MSFT', 'BUY', 295.60, 30, '2025-03-10 11:45:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(97, 'MSFT', 'SELL', 297.30, 15, '2025-03-10 12:00:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(98, 'NVDA', 'BUY', 780.40, 10, '2025-03-10 12:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(99, 'NVDA', 'SELL', 783.50, 10, '2025-03-10 12:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(100, 'TSLA', 'BUY', 660.90, 20, '2025-03-10 12:45:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(101, 'TSLA', 'SELL', 663.15, 5, '2025-03-10 13:00:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(102, 'AAPL', 'BUY', 148.30, 80, '2025-03-10 13:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(103, 'AAPL', 'SELL', 149.20, 40, '2025-03-10 13:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(104, 'GOOG', 'BUY', 2820.55, 18, '2025-03-10 13:45:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(105, 'GOOG', 'SELL', 2840.60, 8, '2025-03-10 14:00:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(106, 'AMZN', 'BUY', 3480.40, 12, '2025-03-10 14:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(107, 'AMZN', 'SELL', 3500.70, 7, '2025-03-10 14:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(108, 'MSFT', 'BUY', 299.10, 22, '2025-03-10 14:45:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(109, 'MSFT', 'SELL', 300.90, 13, '2025-03-10 15:00:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(110, 'NVDA', 'BUY', 790.30, 9, '2025-03-10 15:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(111, 'NVDA', 'SELL', 793.80, 9, '2025-03-10 15:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(112, 'NFLX', 'BUY', 685.20, 28, '2025-03-10 15:45:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(113, 'NFLX', 'SELL', 688.90, 8, '2025-03-10 16:00:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(114, 'TSLA', 'BUY', 655.60, 17, '2025-03-10 16:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(115, 'TSLA', 'SELL', 657.80, 10, '2025-03-10 16:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(116, 'AAPL', 'BUY', 147.00, 75, '2025-03-10 16:45:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(117, 'AAPL', 'SELL', 148.50, 35, '2025-03-10 17:00:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(118, 'GOOG', 'BUY', 2810.40, 20, '2025-03-10 17:15:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(119, 'GOOG', 'SELL', 2830.90, 10, '2025-03-10 17:30:00');
INSERT INTO trade_orders (order_id, symbol, trade_type, price, quantity, trade_date) VALUES 
(120, 'AMZN', 'BUY', 3475.90, 18, '2025-03-10 17:45:00');


update trade_orders 
set trade_flag = 'ABNORMAL'
where quantity > 70;

update trade_orders 
set SLIPPAGE = MARKET_PRICE - PRICE;


INSERT INTO trade_orders (ORDER_ID, TRADE_DATE, SYMBOL, PRICE) VALUES
(86, '2025-03-10 17:45:00', 'AAPL', 150.75),
(87, '2025-03-09 17:45:00', 'GOOGL', 2750.50),
(191, '2025-03-09 14:35:00', 'AAPL', 149.75),
(192, '2025-03-09 14:34:00', 'GOOGL', 2800.40),
(193, '2025-03-09 14:33:00', 'MSFT', 326.10),
(194, '2025-03-09 14:32:00', 'TSLA', 735.55),
(195, '2025-03-09 14:31:00', 'AMZN', 3442.30),
(196, '2025-03-09 14:30:00', 'AAPL', 151.25),
(197, '2025-03-09 14:29:00', 'GOOGL', 2802.50),
(198, '2025-03-09 14:28:00', 'MSFT', 327.40),
(199, '2025-03-09 14:27:00', 'TSLA', 738.75),
(200, '2025-03-09 14:26:00', 'AMZN', 3448.95);
