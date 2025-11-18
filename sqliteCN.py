# use sqllite to store and get data
import sqlite3
import os

def create_database(database_path):
    conn = sqlite3.connect(database_path)
    conn.close()
    return True

def delete_database(database_path):
    if os.path.exists(database_path):
        os.remove(database_path)
        return True
    else:
        return False

def create_table(database_path, table_name, columns):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
    conn.commit()
    conn.close()
    return True


def check_table_exists(database_path, table_name):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None

def delete_table(database_path, table_name):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    conn.close()
    return True

def get_table_column_count(database_path, table_name):
    
    checkTableExists = check_table_exists(database_path, table_name)
    if not checkTableExists:
        return {
            "status": 404,
            "message": "Table not found",
            "count": 0
    }
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    result = cursor.fetchone()
    state = True if result[0] > 0 else False

    if state:
        status = 200
        dataCount = result[0]
    else:
        status = 203
        dataCount = 0
    conn.close()

    replay = {
        "status": status,
        "message": "Table column count",
        "count": dataCount
    }
    return replay