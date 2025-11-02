# GAN Theory Answers

This file contains the answers for **Part 3 – Theory: Building Blocks of GAN** from Assignment 3.  
Repository: https://github.com/milochapman/assignment3-gan

---

### Question 1

Given:
- input size = 8 × 8
- kernel size = 4
- stride = 2
- padding = 1
- output padding = 1

Transposed convolution (2D) formula:
O = (I - 1) * S - 2P + K + OP

Substitute:
I = 8
S = 2
P = 1
K = 4
OP = 1

Compute:
O = (8 - 1) * 2 - 2*1 + 4 + 1
  = 7 * 2 - 2 + 4 + 1
  = 14 - 2 + 4 + 1
  = 17

Answer: output size = 17 × 17.

---

### Question 2

We only change the stride from 2 to 3; all other values stay the same.

Original:
O1 = (I - 1) * 2 - 2P + K + OP

New:
O2 = (I - 1) * 3 - 2P + K + OP

Difference:
O2 - O1 = (I - 1) * (3 - 2)
        = I - 1

Answer: the output spatial size increases by (input_size − 1).
Example: if input = 8, the output increases by 7.

---

### Question 3

General formula for 2D transposed convolution:
O = (I - 1) * S - 2P + K + OP

Where:
- I = input size
- S = stride
- P = padding
- K = kernel size
- OP = output padding

---

### Question 4

Goal: upsample from 16 × 16 to 32 × 32.  
No padding and no output padding, so:
P = 0
OP = 0

Formula becomes:
O = (I - 1) * S + K

Let:
I = 16
O = 32

So:
32 = (16 - 1) * S + K
32 = 15 * S + K

One valid solution:
S = 2
K = 2

Check:
32 = 15*2 + 2 = 30 + 2 = 32

Answer: one valid configuration is
stride = 2, kernel = 2, padding = 0, output padding = 0.
(Other solutions that satisfy 32 = 15*S + K are acceptable.)

---

### Question 5

Mini-batch:
[6, 8, 10, 6]

BatchNorm without learnable parameters (just normalize).

1. Mean
mu = (6 + 8 + 10 + 6) / 4
   = 30 / 4
   = 7.5

2. Variance (BN uses N in the denominator)
var = [ (6 - 7.5)^2 + (8 - 7.5)^2 + (10 - 7.5)^2 + (6 - 7.5)^2 ] / 4
    = (2.25 + 0.25 + 6.25 + 2.25) / 4
    = 11 / 4
    = 2.75

3. Standard deviation
std = sqrt(2.75) ≈ 1.6583

4. Normalize each value
x_hat = (x - mu) / std

6   → (6   - 7.5) / 1.6583 ≈ -0.904
8   → (8   - 7.5) / 1.6583 ≈  0.301
10  → (10  - 7.5) / 1.6583 ≈  1.507
6   → (6   - 7.5) / 1.6583 ≈ -0.904

Answer: normalized batch (3 decimals):
[-0.904, 0.301, 1.507, -0.904]

---

### Question 6

ReLU:
ReLU(x) = max(0, x)

LeakyReLU (negative slope α, for example α = 0.01):
LeakyReLU(x) = x       if x ≥ 0
             = α * x   if x < 0

Key difference:
- ReLU zeroes all negative inputs and has zero gradient there.
- LeakyReLU keeps a small non-zero gradient on the negative side, so backprop can still update weights.

---

### Question 7

Why LeakyReLU is often preferred in GAN discriminators:

1. It prevents “dead” units, because negative inputs still produce gradients.
2. It helps the discriminator give the generator a learning signal at almost every step.
3. It improves stability when activations are not centered.
4. It is standard practice in DCGAN-like architectures.

Answer: LeakyReLU is chosen to keep gradients flowing and to make GAN training more stable.
