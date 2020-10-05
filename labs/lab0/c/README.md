# Lab 0: C codebase  
This folder stores all CUDA C/C++ code for lab 0.  

## Get Started  
```bash
# rectification
# compile
make rectify

# run rectification with /res/Test_1.png and store the output at /c/Test_1_self_rectificated.png with 4 threads
./rectify ../res/Test_1.png ./Test_1_self_rectificated.png 4

# pooling
# compile
make pooling

# run pooling with with /res/Test_1.png and store the output at /c/Test_1_self_pooled.png with 4 threads
./pooling ../res/Test_1.png ./Test_1_self_pooled.png 4

# quality check
# compile tester
make test_equality

# check if Test_1.png is the same as Test_1_self_rectificated.png
./test_equality ../res/Test_1.png ./Test_1_self_pooled.png
```
