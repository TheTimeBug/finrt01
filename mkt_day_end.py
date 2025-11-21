import sqlite3

database_path_adj_data = "finStock.db"
table_name = "mktDayEndData"


def store_mkt_day_end_data_to_sqlite(data):
    # Insert data, update if duplicate based on date and security_code
    conn = sqlite3.connect(database_path_adj_data)
    cursor = conn.cursor()

    # Using INSERT OR REPLACE for upsert functionality
    # This will update the record if a duplicate (based on unique constraint) exists
    sql = """
        INSERT INTO mktDayEndData (
            date, security_code, un_open, un_high, un_low, un_close, un_volume, total_value,
            ad_open, ad_high, ad_low, ad_close, ad_volume, adst_ratio_value, 
            pct_change, pct_change_with_dsex, create_at, update_at
        ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, security_code) DO UPDATE SET
            un_open = excluded.un_open,
            un_high = excluded.un_high,
            un_low = excluded.un_low,
            un_close = excluded.un_close,
            un_volume = excluded.un_volume,
            total_value = excluded.total_value,
            ad_open = excluded.ad_open,
            ad_high = excluded.ad_high,
            ad_low = excluded.ad_low,
            ad_close = excluded.ad_close,
            ad_volume = excluded.ad_volume,
            adst_ratio_value = excluded.adst_ratio_value,
            pct_change = excluded.pct_change,
            pct_change_with_dsex = excluded.pct_change_with_dsex,
            create_at = excluded.create_at,
            update_at = excluded.update_at
    """

    cursor.executemany(sql, data)
    conn.commit()
    conn.close()

    reply_data = {
        "status": 200,
        "message": "Data stored/updated successfully",
        "data_count": len(data),
    }
    return reply_data


def post_mkt_day_end_data_to_sqlite(marketDayEndData: list):
    try:
        batch_data_list = []
        for item in marketDayEndData:
            batch_data_list.append(
                (
                    item["date"],
                    item["security_code"],
                    item["un_open"],
                    item["un_high"],
                    item["un_low"],
                    item["un_close"],
                    item["un_volume"],
                    item["total_value"],
                    item["ad_open"],
                    item["ad_high"],
                    item["ad_low"],
                    item["ad_close"],
                    item["ad_volume"],
                    item["adst_ratio_value"],
                    item["pct_change"],
                    item["pct_change_with_dsex"],
                    item["create_at"],
                    item["update_at"],
                )
            )
        result = store_mkt_day_end_data_to_sqlite(batch_data_list)
        if result["status"] == 200:
            reply_data = {
                "status": 200,
                "message": "Data stored successfully",
                "data_count": len(batch_data_list),
            }
        else:
            reply_data = {
                "status": 500,
                "message": "Data stored failed",
                "data_count": len(batch_data_list),
            }
        return reply_data
    except Exception as e:
        return {"status": 500, "message": f"Error processing data: {str(e)}"}


def get_mkt_day_end_data_from_sqlite(filter_type: str = None, filter_value: str = None):
    conn = sqlite3.connect(database_path_adj_data)
    cursor = conn.cursor()

    if filter_type and filter_value:
        if filter_type == "date":
            sql = "SELECT * FROM mktDayEndData WHERE date = ? ORDER BY date DESC"
            cursor.execute(sql, (filter_value,))
        elif filter_type == "security_code":
            sql = (
                "SELECT * FROM mktDayEndData WHERE security_code = ? ORDER BY date DESC"
            )
            cursor.execute(sql, (filter_value,))
        elif filter_type == "limit":
            sql = "SELECT * FROM mktDayEndData ORDER BY date DESC LIMIT ?"
            cursor.execute(sql, (int(filter_value),))
        else:
            sql = "SELECT * FROM mktDayEndData ORDER BY date DESC LIMIT 300"
            cursor.execute(sql)
    else:
        sql = "SELECT * FROM mktDayEndData ORDER BY date DESC LIMIT 300"
        cursor.execute(sql)

    result = cursor.fetchall()
    conn.close()
    return result


def get_today_mkt_day_end_data_from_sqlite():
    conn = sqlite3.connect(database_path_adj_data)
    conn.row_factory = sqlite3.Row  # Enable column names
    cursor = conn.cursor()
    sql = "SELECT security_code, ad_close as market_price FROM mktDayEndData WHERE date = ? ORDER BY date ASC"
    # today_date = datetime.now().strftime("%Y-%m-%d")
    today_date = "2025-11-02"
    cursor.execute(sql, (today_date,))
    rows = cursor.fetchall()
    conn.close()

    # Convert sqlite3.Row objects to dictionaries
    result = [dict(row) for row in rows]
    return result


def get_historical_market_price(security_codes):
    # Convert numpy array to list if needed
    if hasattr(security_codes, "tolist"):
        security_codes = security_codes.tolist()
        # add 00DSEX to the list
        security_codes.append("00DSEX")

    conn = sqlite3.connect(database_path_adj_data)
    conn.row_factory = sqlite3.Row  # Enable column names
    cursor = conn.cursor()

    # Create placeholders for parameterized query (safer than f-string)
    placeholders = ",".join(["?" for _ in security_codes])
    sql = f"SELECT security_code, date, ad_close as market_price,ad_open,ad_high,ad_low,ad_volume,ad_close FROM mktDayEndData WHERE security_code IN ({placeholders}) ORDER BY date ASC"

    cursor.execute(sql, security_codes)
    rows = cursor.fetchall()
    conn.close()
    # Convert sqlite3.Row objects to dictionaries
    result = [dict(row) for row in rows]
    return result


# get all security codes last valid market price not null or zero
def get_all_security_codes_last_valid_market_price():
    conn = sqlite3.connect(database_path_adj_data)
    conn.row_factory = sqlite3.Row  # Enable column names
    cursor = conn.cursor()
    sql = "SELECT security_code, ad_close AS market_price, date FROM ( SELECT security_code, ad_close, date, ROW_NUMBER() OVER ( PARTITION BY security_code ORDER BY date DESC ) AS rn FROM mktDayEndData WHERE ad_close IS NOT NULL AND ad_close != 0 ) AS x WHERE rn = 1;"
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_last_30_days_market_value_avg(security_codes: list):
    remove_00DSEX = [code for code in security_codes if code != "00DSEX"]
    placeholders = ",".join(["?" for _ in remove_00DSEX])

    conn = sqlite3.connect(database_path_adj_data)
    conn.row_factory = sqlite3.Row  # Enable column names
    cursor = conn.cursor()
    sql = f"SELECT security_code, AVG(total_value) AS avg_total_value FROM ( SELECT *, ROW_NUMBER() OVER (PARTITION BY security_code ORDER BY date DESC) AS rn FROM {table_name} WHERE security_code IN ({placeholders}) ) t WHERE rn <= 30 GROUP BY security_code;"
    cursor.execute(sql, remove_00DSEX)
    adtv = cursor.fetchall()
    conn.close()
    return adtv
