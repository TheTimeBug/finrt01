import sqlite3

database_path_adj_data = "finStock.db"
table_name = "mktDayEndData"
api_url_adj_data = "https://fi-rt.cottonstock.com/api_adj_data_all.php"
api_url_adj_data_new = "https://fi-rt.cottonstock.com/api_adj_data_new.php"

# Authentication token for API requests
AUTH_TOKEN = "Bearer c1uYPdG29rjlqgWJ6Dy3u8jhsdkusahdask"


def get_auth_headers():
    """
    Get authentication headers for API requests
    Returns headers dict with Authorization token
    """
    return {
        "Authorization": AUTH_TOKEN,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def store_mkt_day_end_data_to_sqlite(data):
    # Insert data, update if duplicate based on date and security_code
    conn = sqlite3.connect(database_path_adj_data)
    cursor = conn.cursor()

    # Using INSERT OR REPLACE for upsert functionality
    # This will update the record if a duplicate (based on unique constraint) exists
    sql = """
        INSERT INTO mktDayEndData (
            date, security_code, un_open, un_high, un_low, un_close, un_volume, 
            ad_open, ad_high, ad_low, ad_close, ad_volume, adst_ratio_value, 
            pct_change, pct_change_with_dsex, create_at, update_at
        ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, security_code) DO UPDATE SET
            un_open = excluded.un_open,
            un_high = excluded.un_high,
            un_low = excluded.un_low,
            un_close = excluded.un_close,
            un_volume = excluded.un_volume,
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
