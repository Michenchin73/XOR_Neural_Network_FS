# A Neural Network from Scratch
## No AI libraries only math and random

Here is the architecture of the Neural Network
![](./img/Diagrama%20sin%20título.png)

$$N_1 = x_1 w_1 + x_2 w_2 + b_1$$
$$N_2 = x_1 w_3 + x_2 w_4 + b_2$$
$$v_1 = \sigma (N1)$$
$$v_2 = \sigma (N2)$$
$$h_1 = v_1 w_5 + v_2 w_6 + b_3$$
$$h_2 = v_1 w_7 + v_2 w_8 + b_4$$
$$z_1 = \sigma (h_1)$$
$$z_2 = \sigma (h_2)$$
$$z_3 = z_1 w_9 + z_2 w_{10} + b_5$$
$$\boxed{\hat{y} = \sigma (z_3)}$$

> [!NOTE]
> In my previous project "linear_regressionFS" in docs there's a LaTeX pdf where this is explained in WAY MORE detail. Check it if you want to go deeper!!

You can check my previous project [HERE](https://github.com/Michenchin73/linear_regresionFS)
