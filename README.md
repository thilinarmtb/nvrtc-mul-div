## Build Program

```sh
mkdir build; cd build; cmake ..; make; cd -
```

## Run Program

```sh
[thilina@login]$ ./build/vec 0 5000000
div selected.
CUDA kernel launch with 19532 blocks of 256 threads
Time = 0.000075
[thilina@login]$ ./build/vec 1 5000000
mul selected.
CUDA kernel launch with 19532 blocks of 256 threads
Time = 0.000076
```
