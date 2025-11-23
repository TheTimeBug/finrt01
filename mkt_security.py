import sqlite3

database_path_security = "finStock.db"
table_name = "mktSecurityInfo"


def store_mkt_security_to_sqlite(data):
    conn = sqlite3.connect(database_path_security)
    cursor = conn.cursor()

    # Check and update existing records, or insert new ones
    # Since table doesn't have UNIQUE constraint, we need to handle upsert manually
    for record in data:
        security_code = record[1]  # security_code is at index 1

        # Check if record exists
        check_sql = f"SELECT id FROM {table_name} WHERE security_code = ?"
        cursor.execute(check_sql, (security_code,))
        existing = cursor.fetchone()

        if existing:
            # Update existing record
            update_sql = f"""
                UPDATE {table_name} SET
                    is_active = ?,
                    company_name = ?,
                    sector = ?,
                    marginable = ?,
                    un_close = ?,
                    ad_close = ?,
                    volitility = ?,
                    betas = ?,
                    outstanding_shares_mn = ?,
                    floating_percent = ?,
                    floating_shares_mn = ?,
                    create_at = ?,
                    update_at = ?
                WHERE security_code = ?
            """
            cursor.execute(
                update_sql,
                (
                    record[0],
                    record[2],
                    record[3],
                    record[4],
                    record[5],
                    record[6],
                    record[7],
                    record[8],
                    record[9],
                    record[10],
                    record[11],
                    record[12],
                    record[13],
                    security_code,
                ),
            )
        else:
            # Insert new record
            insert_sql = f"""
                INSERT INTO {table_name} (
                    is_active, security_code, company_name, sector, marginable,
                    un_close, ad_close, volitility, betas, outstanding_shares_mn,
                    floating_percent, floating_shares_mn, create_at, update_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_sql, record)
    conn.commit()
    conn.close()

    reply_data = {
        "status": 200,
        "message": "Security data stored/updated successfully",
    }
    return reply_data


def post_mkt_security_to_sqlite(securityData: list):
    try:
        batch_data_list = []
        for item in securityData:
            batch_data_list.append(
                (
                    item["is_active"],
                    item["security_code"],
                    item["company_name"],
                    item["sector"],
                    item["marginable"],
                    item["un_close"],
                    item["ad_close"],
                    item["volitility"],
                    item["betas"],
                    item.get("outstanding_shares_mn"),  # Can be NULL
                    item.get("floating_percent"),  # Can be NULL
                    item.get("floating_shares_mn"),  # Can be NULL
                    item["create_at"],
                    item["update_at"],
                )
            )
        result = store_mkt_security_to_sqlite(batch_data_list)
        if result["status"] == 200:
            reply_data = {
                "status": 200,
                "message": "Security data stored successfully",
                "data_count": len(batch_data_list),
            }
        else:
            reply_data = {
                "status": 500,
                "message": "Security data stored failed",
                "data_count": len(batch_data_list),
            }
        return reply_data
    except Exception as e:
        return {"status": 500, "message": f"Error processing security data: {str(e)}"}


def get_mkt_security_from_sqlite():
    conn = sqlite3.connect(database_path_security)
    cursor = conn.cursor()
    sql = f"SELECT * FROM {table_name}"
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()
    return result

def get_outstanding_shares_security_codes():
    conn = sqlite3.connect(database_path_security)
    cursor = conn.cursor()
    sql = f"SELECT security_code, outstanding_shares_mn, floating_shares_mn ,floating_percent FROM {table_name}"
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()
    return result