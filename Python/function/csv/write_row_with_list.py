import csv
import pandas as pd


def write_csv_dynamically(csv_file_path, csv_file_factor_name_list, rows):
    
    # Write the factor names.
    csv_file = open(csv_file_path, "w", newline = "")
    writer = csv.writer(csv_file)
    writer.writerow(csv_file_factor_name_list)
    csv_file.close()
    
    # Write rows.
    for row_in_list in rows:
        csv_file = open(csv_file_path, "a", newline = "")
        writer = csv.writer(csv_file)
        writer.writerow(row_in_list)
        csv_file.close()


def write_csv_all_at_once(csv_file_path, csv_file_factor_name_list, rows):
    
    with open(csv_file_path, "w", newline = "") as csv_file:
        
        writer = csv.writer(csv_file)
        
        # Write the factor names.
        writer.writerow(csv_file_factor_name_list)
        
        # Write rows.
        for row_in_list in rows:
            writer.writerow(row_in_list)


def main():
    
    # enable_to_observe_csv_at_any_time = True
    enable_to_observe_csv_at_any_time = False
    
    csv_file_path = "test_list.csv"
    csv_file_factor_name_list = ["column_1", "column_2", "column_empty", "column_3"]
    rows = []
    for i in range(10):
        rows.append(["column_1_" + str(i + 1), \
                     "column_2_" + str(i + 1), None, \
                     "column_3_" + str(i + 1)])
    
    if enable_to_observe_csv_at_any_time == True:
        write_csv_dynamically(csv_file_path, csv_file_factor_name_list, rows)
    elif enable_to_observe_csv_at_any_time == False:
        write_csv_all_at_once(csv_file_path, csv_file_factor_name_list, rows)
    
    
    df = pd.read_csv(csv_file_path)
    print(df)


if __name__ == "__main__":
    
    main()