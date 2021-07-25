# mandelbrot_viewer
Linux dependencies:
+ build-essential;
+ libglu1-mesa-dev;
+ libpng-dev.

Compile and run:
```bash
g++-10 -O3 -o main main.cpp -march=native -lX11 -lGL -lpthread -lpng -lstdc++fs -std=c++17 && ./main
```

Controls:

```
Q                       - Zoom in
A                       - Zoom out
Num keys (NOT keypad) 0 - Reset screen to starting position; 1 to 6 - Choose rendering method
Left arrow key          - Decrement number of used threads
Right arrow key         - Increment number of used threads
Up arrow key            - Increase maximum iterations of the algorithm
Down arrow key          - Decrease maximum iterations of the algorithm
```

olcPixelGameEngine: https://github.com/OneLoneCoder/olcPixelGameEngine
