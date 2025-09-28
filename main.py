def get_sum(a: float, b: float, c: float) -> float:
    return a + b + c


if __name__ == '__main__':
    num1, num2, num3 = map(float, input().split())
    res = get_sum(num1, num2, num3)
    print(res)
