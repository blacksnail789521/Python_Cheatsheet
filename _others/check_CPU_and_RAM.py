import psutil

cpu_percent = psutil.cpu_percent()
ram_percent = psutil.virtual_memory().percent

print(f"CPU percent: {cpu_percent}")
print(f"RAM percent: {ram_percent}")