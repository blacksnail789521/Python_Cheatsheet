class print_with_fstring():
    
    def __init__(self):
        
        self.global_var = "global"
    
    
    def print_var(self):
        
        local_val = "local"
        
        print(f"global_var: {self.global_var}")
        print(f"local_val: {local_val}")


print_with_fstring().print_var()