def get_sum(*args: float) -> float:
    return sum(args)


if __name__ == '__main__':
    nums = map(float, input().split())
    res = get_sum(*nums)
    print(res)
