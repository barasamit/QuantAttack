# 8_bits_attack

ADDS to site packages

1.

Go to your site package of /transformers/bitsandbytes/

"/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/transformers/bitsandbytes/"  
  
Add to function replace_8bit_linear the following code in blue,

Change the root_path to yours.  
  
![](media/fb4b5e11fecfb97e7481d584f25df13d.png)

2.

Go to your site package of bitsandbytes/autograd/_functions.py

"/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/bitsandbytes/autograd/_functions.py"  
  
Add to Class MatMul8bitLt forward function the following code in blue,

Change the root_path to yours.

![](media/cf79840d8e464b443ede4342481bc7b1.png) ![](media/f6871591af2c0f21f1161a2bbf847c34.png)
