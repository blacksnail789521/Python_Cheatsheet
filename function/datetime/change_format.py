from datetime import datetime


# Change from string to datetime.
string_form = "2020-12-01 08:09:10.234"
datetime_form = datetime.strptime(string_form, "%Y-%m-%d %H:%M:%S.%f")

# Change from datetime to string.
new_string_form = datetime_form.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # The only way to get three-digit milliseconds.

assert string_form == new_string_form