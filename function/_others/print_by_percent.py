def print_percent(current_percent, index, show_index):
    
    if show_index == False:
        print(f"{current_percent:3} %")
    else:
        print(f"{current_percent:3} % (index: {index})")

def print_by_percent(index, total_length, percent = 10, show_index = False):
    
    current_percent = 100 * index // total_length
    current_percent = current_percent // percent * percent
    next_percent = 100 * (index + 1) // total_length
    next_percent = next_percent // percent * percent
    while current_percent < next_percent:
        print_percent(current_percent, index, show_index)
        current_percent += percent
    
    
    # Show the remaining percent.
    if index == total_length - 1:
        while current_percent <= 100:
            print_percent(current_percent, index, show_index)
            current_percent += percent
            # 100 is not divisible by percent, so we need to show "100 %" by ourselves.
            if current_percent > 100 and current_percent - percent != 100:
                print_percent(100, index, show_index)


length = 250
iterable = list(range(0, length))
for index, element in enumerate(iterable):
    print_by_percent(index, len(iterable), percent = 8, show_index = True)