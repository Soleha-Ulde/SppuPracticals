def recursive_fib(n):
    if n<=1:
        return n
    else :
        return recursive_fib(n-1)+recursive_fib(n-2)
    
def non_recursive_fib(n):
    a = 0
    b = 1
    print(a)
    print(b)
    for i in range(2,n):
        c = a + b
        print(c)
        a = b
        b = c
print("Fibonacci using Recursive approach")
print(recursive_fib(0))
print(recursive_fib(1))
print(recursive_fib(2))
print(recursive_fib(3))
print(recursive_fib(4))
print("Fibonnacci Series using non-recursive approach")
non_recursive_fib(8)