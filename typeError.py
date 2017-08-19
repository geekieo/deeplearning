class TypeError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def test():
    # 1
    try:
        raise TypeError(2*2)
    except TypeError as e:
        print("Type exception occurred, value:", e.value)

    # 2
    raise TypeError('check the type of your value!')
    
