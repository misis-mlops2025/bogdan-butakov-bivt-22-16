def get_sum(a: float, b: float) -> float:
    return a + b


if __name__ == '__main__':
    num1, num2 = map(float, input().split())
    res = get_sum(num1, num2)
    print(res)
