# 8_bits_attack

## run:
1. one_to_one_attack.py
2. many_to_many_attack.py


## Adds to site packages:

1.

Go to your site package of /transformers/bitsandbytes/

"/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/transformers/bitsandbytes/"  
  
Add to function replace_8bit_linear the following code in blue,

Change the root_path to yours.  
  
![image](https://user-images.githubusercontent.com/96978735/232737835-242a8483-9754-437f-9ef6-80904fba46ef.png)
2.

Go to your site package of bitsandbytes/autograd/_functions.py

"/sise/home/barasa/.conda/envs/bnb/lib/python3.10/site-packages/bitsandbytes/autograd/_functions.py"  
  
Add to Class MatMul8bitLt forward function the following code in blue,

Change the root_path to yours.

![image](https://user-images.githubusercontent.com/96978735/232738006-461bdacc-b9bc-4390-9731-8b7b28b1108b.png)
