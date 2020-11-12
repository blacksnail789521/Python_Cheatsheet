from DBSrv5ConnPy.db_object import DBObject

FAB = "F12"

# Initialize the sql for getting recipe_table.
db_obj = DBObject(fab = FAB, db = "MES")
sql = \
    """
    DELETE FROM MFGDEV3.AUTO_MINE_PIPELINE_LOG_BTH
    WHERE AP_NAME LIKE 'some_module%'
    """
db_obj.non_query(sql = sql)