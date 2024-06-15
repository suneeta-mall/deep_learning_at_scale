def make_add_x(x):
    def add_x(y):
        return x + y

    return add_x


add_4 = make_add_x(4)
add_5 = make_add_x(5)
