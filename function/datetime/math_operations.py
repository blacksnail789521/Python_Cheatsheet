from datetime import datetime, timedelta


current_time = datetime.now()
delta = timedelta(weeks = 1, days = 1, hours = 1, minutes = 1, seconds = 1, milliseconds = 10, microseconds = 9)
future_time = current_time + delta

print(current_time)
print("           +")
print(delta)
print("           +")
print(future_time)