import sqlite3

database_path = "finStock.db"

def get_last_entry_from_sqlite():
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT max(date),max(create_at),max(update_at) FROM mktDayEndData")
    result1 = cursor.fetchone()
    

    cursor.execute("SELECT max(create_at),max(update_at) FROM mktSecurityInfo")
    result2 = cursor.fetchone()
    conn.close()

    if result1 and result2:
        mded_date = result1[0]
        mded_create_at = result1[1]
        mded_update_at = result1[2]
        msi_create_at = result2[0]
        msi_update_at = result2[1]
        result = {
            "status": 200,
            "message": "Last entry fetched successfully",
            "market_day_end_data": {
                "mded_date": mded_date,
                "mded_create_at": mded_create_at,
                "mded_update_at": mded_update_at,
            },
            "market_security_info": {
                "msi_create_at": msi_create_at,
                "msi_update_at": msi_update_at,
            },
        }
    else:
        result = {
            "status": 500,
            "message": "Last entry not found",
        }
    return result