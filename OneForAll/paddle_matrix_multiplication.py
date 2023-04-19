import paddle


print(paddle.utils.run_check())

x = paddle.randn([5, 3])
print(x)

y = paddle.ones([3, 5])
print(y)

z = paddle.matmul(x, y)
print(z)
