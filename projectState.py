from sqliteCN import (
    check_table_exists,
    create_database,
    create_table,
    delete_database,
    delete_table,
)

database_path = "finStock.db"
mktDayEndData_table_name = "mktDayEndData"
mktSecurityInfo_table_name = "mktSecurityInfo"


# initialize the project state
def projectState_init():
    result0 = create_database(database_path)
    result1 = create_MktDayEndData_table_if_not_exists()
    result2 = create_MktSecurityInfo_table_if_not_exists()
    if result0 and result1 and result2:
        return True
    else:
        return False


# delete the project state
def projectState_delete():
    result1 = delete_MktDayEndData_table_if_not_exists()
    result2 = delete_MktSecurityInfo_table_if_not_exists()
    result3 = delete_database(database_path)
    if result1 and result2 and result3:
        return True
    else:
        return False


########################## Helper Functions ##########################
######################################################################


# delete the mkt day end data table if it exists
def delete_MktDayEndData_table_if_not_exists():
    if check_table_exists(database_path, mktDayEndData_table_name):
        result = delete_table(database_path, mktDayEndData_table_name)
        if result:
            print("Table deleted successfully")
            return True
        else:
            print("Table deletion failed")
            return False
    else:
        print("Table not found")
        return True


# delete the mkt security info table if it exists
def delete_MktSecurityInfo_table_if_not_exists():
    if check_table_exists(database_path, mktSecurityInfo_table_name):
        result = delete_table(database_path, mktSecurityInfo_table_name)
        if result:
            print("Table deleted successfully")
            return True
        else:
            print("Table deletion failed")
            return False
    else:
        print("Table not found")
        return True


######################################################################
# create functions ########################################################
######################################################################


# create the mkt day end data table if it doesn't exist
def create_MktDayEndData_table_if_not_exists():
    if check_table_exists(database_path, mktDayEndData_table_name):
        print("Table already exists")
        return True
    else:
        adjTableColumn = "id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE NOT NULL, security_code VARCHAR(32) NOT NULL, un_open FLOAT, un_high FLOAT, un_low FLOAT, un_close FLOAT, un_volume INTEGER, total_value FLOAT, ad_open FLOAT, ad_high FLOAT, ad_low FLOAT, ad_close FLOAT, ad_volume INTEGER, adst_ratio_value FLOAT, pct_change FLOAT, pct_change_with_dsex FLOAT, create_at DATETIME, update_at DATETIME, UNIQUE(date, security_code)"
        result = create_table(database_path, mktDayEndData_table_name, adjTableColumn)
        if result:
            print("Table created successfully")
            return True
        else:
            print("Table creation failed")
            return False


# create the mkt security info table if it doesn't exist
def create_MktSecurityInfo_table_if_not_exists():
    if check_table_exists(database_path, mktSecurityInfo_table_name):
        print("Table already exists")
        return True
    else:
        mktSecurityInfoTableColumn = (
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "is_active CHAR(1) NOT NULL, "
            "security_code VARCHAR(64) NOT NULL, "
            "company_name VARCHAR(128) NOT NULL, "
            "sector VARCHAR(64) NOT NULL, "
            "marginable CHAR(1) NOT NULL, "
            "un_close FLOAT NOT NULL, "
            "ad_close FLOAT NOT NULL, "
            "volitility FLOAT NOT NULL, "
            "betas FLOAT NOT NULL, "
            "outstanding_shares_mn FLOAT, "
            "floating_percent FLOAT, "
            "floating_shares_mn FLOAT, "
            "create_at DATETIME NOT NULL, "
            "update_at DATETIME NOT NULL"
        )
        result = create_table(
            database_path, mktSecurityInfo_table_name, mktSecurityInfoTableColumn
        )
        if result:
            print("Table created successfully")
            return True
        else:
            print("Table creation failed")
            return False
