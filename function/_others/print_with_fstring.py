class print_with_fstring():
    
    def __init__(self):
        
        self.global_var = "global"
    
    
    def print_var(self):
        
        local_var = "local"
        
        print(f"global_var: {self.global_var}")
        print(f"local_var: {local_var}")


print_with_fstring().print_var()