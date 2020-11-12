full_message = \
    """
    2019-10-31 10:14:13.286 | INFO  | some_module               : test_function        : 12   | 0
    2019-10-31 10:14:13.286 | INFO  | some_module               : test_function        : 12   | 1
    2019-10-31 10:14:13.286 | INFO  | some_module               : test_function        : 12   | 2
    2019-10-31 10:14:13.286 | INFO  | some_module               : test_function        : 12   | 3
    2019-10-31 10:14:13.286 | INFO  | some_module               : test_function        : 12   | 4
    2019-10-31 10:14:13.286 | INFO  | some_module               : test_function        : 12   | 5
    2019-10-31 10:14:13.286 | INFO  | some_module               : test_function        : 12   | 6
    2019-10-31 10:14:13.287 | INFO  | some_module               : test_function        : 12   | 7
    
    """


message_list = str(full_message).split("\n")

sql = \
    """
    INSERT INTO MFGDEV3.AUTO_MINE_TOOL_RCP_BT
    (EQP_ID, REMARK)
    VALUES
    """
sql_with_single_row = \
    """
    ('{EQP_ID}', q'{{{REMARK}}}')
    """
for index, message in enumerate(message_list):
    sql = sql + sql_with_single_row.format(EQP_ID = "test_" + "{:03d}".format(index), \
                                           REMARK = message)
    if index != len(message_list) - 1:
        sql = sql + ","

print(sql)