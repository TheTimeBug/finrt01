import sqlite3
import time

import requests

from sqliteCN import get_table_column_count

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


def api_adj_data_all():
    limit = 100
    iniStart = 0
    total_records = 0

    try:
        state = get_table_column_count(database_path_adj_data, table_name)
        if state["status"] == 404:
            return state
        else:
            start = state["count"]
            iniStart = start

            while True:
                api_url_with_params = f"{api_url_adj_data}?start={start}&limit={limit}"
                print(f"Fetching from: {api_url_with_params}")

                # Retry logic for connection issues
                max_retries = 3
                retry_count = 0
                response = None

                while retry_count < max_retries:
                    try:
                        # Add Authorization header to request
                        headers = get_auth_headers()
                        response = requests.get(
                            api_url_with_params, headers=headers, timeout=60
                        )
                        break  # Success, exit retry loop
                    except requests.exceptions.ConnectionError as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = retry_count * 2  # Exponential backoff
                            print(
                                f"Connection failed. Retry {retry_count}/{max_retries} in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            return {
                                "status": 500,
                                "error": "Connection error after retries",
                                "message": f"Failed after {max_retries} attempts. Remote server closed connection.",
                                "api_url": api_url_with_params,
                                "details": str(e),
                                "records_synced_before_error": total_records,
                                "suggestion": "Check if PHP script is working: curl the URL directly, check PHP error logs, increase PHP timeout/memory",
                            }

                if response is None or response.status_code != 200:
                    if response:
                        print(f"API returned status code: {response.status_code}")
                    break

                try:
                    responseData = response.json()
                    batch_data_list = []

                    for item in responseData:
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

                    if len(batch_data_list) > 0:
                        store_mkt_day_end_data_to_sqlite(batch_data_list)
                        total_records += len(batch_data_list)
                        print(
                            f"Stored {len(batch_data_list)} records. Total: {total_records}"
                        )
                    else:
                        break

                    if len(batch_data_list) < limit or total_records >= 1300:
                        break

                    start += limit

                except requests.exceptions.Timeout:
                    return {
                        "status": 504,
                        "error": "Timeout",
                        "message": "API request timed out after 60 seconds",
                        "api_url": api_url_with_params,
                        "records_synced_before_error": total_records,
                    }
                except ValueError as e:
                    return {
                        "status": 500,
                        "error": "Invalid JSON",
                        "message": "API response is not valid JSON",
                        "api_url": api_url_with_params,
                        "details": str(e),
                        "records_synced_before_error": total_records,
                    }

        return {
            "status": 200,
            "message": "Data synced successfully",
            "total_records_synced": total_records,
            "batches_processed": (iniStart // limit) + 1,
        }

    except Exception as e:
        return {
            "status": 500,
            "error": "Unexpected error",
            "message": str(e),
            "error_type": type(e).__name__,
            "records_synced_before_error": total_records,
        }


def store_mkt_day_end_data_to_sqlite(data):
    # check if table exists
    conn = sqlite3.connect(database_path_adj_data)
    cursor = conn.cursor()
    sql = "INSERT INTO adj_data (date, security_code, un_open, un_high, un_low, un_close, un_volume, ad_open, ad_high, ad_low, ad_close, ad_volume, adst_ratio_value, pct_change, pct_change_with_dsex, create_at, update_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    cursor.executemany(sql, data)
    conn.commit()
    conn.close()
    reply_data = {"message": "Data stored successfully", "data_count": len(data)}
    return reply_data
