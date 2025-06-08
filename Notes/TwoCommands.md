A single token with an embedding dimension of 512 at half precision takes up 1024 bytes or 1 Kilobyte (KB) of memory.

At image size of 128, that's 


Pixel = Quaternion Value
Upsample to 128 * 4 quaternions at 512d each
Simple 

Input Values - Minimal * Type Upscaler (Color, Character, Word, Number)

Value Token * Type Token = Input Token

Standard attention


GPT2-XL
Parameters:                             ~1.5 billion (1,542M)
Embedding Dimension (d_model):          1600
Number of Layers (n_layer):             48
Number of Attention Heads (n_head):     25
Context Window:                         1024 tokens
Vocabulary Size:                        50,257

Image > Code
Code x2 (t1, t2) > Pair Code
Pair Code xN (Ex. 1, Ex. 2, ..., Ex. N) > Set Code
Set Code + Test Input > Test Output

SCOPE IS TOO BIG ↴ ↴ ↴ 

There should only be 2 commands.

1. + New (Create, Write)
2. View (Shift, Go to, Navigate, Traverse, Read)

We provide the model with source documents or files, as well as a "space" within 
which it can navigate and create its own.

It comes with a built in scratchpad in VRAM where it can store thoughts or notes 
that it deems important. It can collect files, or portions of files to run through 
the attention mechanism. At the end of the attention layers it can either output 
or send a new group of tokens back through.

For example it receives images that are too large to process. So it naively breaks 
the image into a grid of "patches" and processes each of those and takes notes on 
the output. Then let's say an image has an object that straddles two patches, or 
three. From there it has learned to go back and look at that area creating a new
patch that has different portions of the old patch. It can zoom in for more detail,
it can slide it's viewport around and see what is beneficial to look at. It can 
trace an edge and count the pixels of each color value. Etc.

The model needs to be able to go to view distinct files, traverse up and down 
their AST, look at things together, and in isolation, etc.

We replace, “Desktop > Directory > File,” with, “Hyperdoc > Block > Tag.”
We replace, “echo, touch, ls, mkdir, grep, man, pwd, cd, mv, rmdir, locate, >, 
|, head, tail, chmod, exit, history, clear, cp, kill, and sleep,” with 
“new, and view.”

And the model navigates this abstract and also locally spatial 4D hyperspace.