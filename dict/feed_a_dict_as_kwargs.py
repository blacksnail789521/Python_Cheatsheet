def test(a: str = "1", b: int = 2):
    print(a, b)


test(**{"a": "XD", "b": 3})  # XD 3
d = {"a": "0"}
test(**d)  # 0 2
