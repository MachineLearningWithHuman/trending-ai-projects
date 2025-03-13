# Trade Order Management System

This project is a Trade Order Management System built using Snowflake, Snowpark, and Streamlit. The system allows for the management and analysis of trade orders, including real-time processing, historical data retrieval, and visualization.

## Features

- **Trade Order Table**: Stores trade orders with details such as symbol, trade type, quantity, price, and status.
- **Stream and Task**: Automatically detects and processes new trade orders.
- **Historical Data Retrieval**: Uses Snowpark Stored Procedures to fetch historical trade data using Snowflake's Time Travel feature.
- **Streamlit Dashboard**: Provides a real-time and historical trade analysis dashboard.

## Files Overview

### `app.py`
This is the Streamlit application that provides a dashboard for live and historical trade analysis. It includes:
- Real-time display of the latest trade orders.
- Historical trade analysis with visualizations such as trade volume over time, stock price movements, and trade type distribution.
- Advanced trading insights including volatility analysis, order flow imbalance, and moving average trends.

### `example.sql`
This SQL script sets up the initial database, schema, and `trade_orders` table. It also includes:
- Sample data insertion.
- Time Travel queries to fetch historical data.
- Stream and Task creation for processing new trade orders.

### `python stored procedure output viz.sql`
This script contains a Snowpark Python stored procedure that fetches historical trade data using Time Travel and performs basic analysis on flagged trades.

### `sql script.sql`
This script includes:
- Database and schema creation.
- Table and stream setup.
- Task creation for processing trade orders.
- Sample data insertion and updates to simulate market conditions.

### `stored procedures.sql`
This script defines two Snowpark stored procedures:
- `get_historical_trades`: Fetches historical trade data using Time Travel.
- `get_trade_analysis`: Analyzes trade execution delays, slippage, and anomalies.

## Setup and Usage

### Prerequisites
- Snowflake account with necessary permissions.
- Streamlit installed locally or deployed on a cloud service.
- Snowpark Python package installed.

  [View on Eraser![](https://app.eraser.io/workspace/yxJTpPtTwgOevVzESdLj/preview?elements=N-BcHO835GmqLdoL9eVfPA&type=embed)](https://app.eraser.io/workspace/yxJTpPtTwgOevVzESdLj?elements=N-BcHO835GmqLdoL9eVfPA)

### Steps to Run the Project

1. **Database and Table Setup**:
   - Run the `example.sql` script to create the database, schema, and `trade_orders` table.
   - Insert sample data using the provided SQL commands.

2. **Stream and Task Setup**:
   - Create a stream and task to process new trade orders as shown in `example.sql`.

3. **Stored Procedures**:
   - Execute the `stored procedures.sql` script to create the necessary stored procedures for historical data retrieval and trade analysis.

4. **Streamlit App**:
   - Run the `app.py` script using Streamlit to start the dashboard.
   - Use the dashboard to view real-time trade orders and perform historical trade analysis.

## Visualizations

The Streamlit app provides several visualizations, including:
- **Trade Volume Over Time**: Line chart showing the total trade volume over time.
- **Stock Price Movements**: Line chart displaying price movements for different stocks.
- **Trade Type Distribution**: Bar chart showing the distribution of buy and sell trades.
- **Stock-wise Trade Volume**: Bar chart displaying trade volume by stock symbol.
- **Advanced Insights**: Includes volatility analysis, order flow imbalance, and moving average trends.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Snowflake for providing the platform and tools.
- Streamlit for the easy-to-use dashboard framework.
- Snowpark for enabling Python-based stored procedures and data processing.

---

For any questions or issues, please open an issue in the repository.
