import numpy as np

# shift right
def sftR (x, n):
    x >>= n
    return x

# rotate right
def rotR (x, n):
    for _ in range (n):
        if x % 2: # if last bit is 1
            x >>= 1 
            x += 2**31
        else: x = x >> 1
    return x

def xor (x, y):
    return np.bitwise_xor(x, y)

def sigma0 (x):
    a = rotR (x, 7)
    b = rotR (x, 18)
    c = xor (a, b)
    d = sftR (x, 3)
    return xor (c, d)

def sigma1 (x):
    a = rotR (x, 17)
    b = rotR (x, 19)
    c = xor (a, b)
    d = sftR (x, 10)
    return xor (c, d)

def SIGMA0 (x):
    a = rotR (x, 2)
    b = rotR (x, 13)
    c = xor (a, b)
    d = sftR (x, 22)
    return xor (c, d)

def SIGMA1 (x):
    a = rotR (x, 6)
    b = rotR (x, 11)
    c = xor (a, b)
    d = sftR (x, 25)
    return xor (c, d)

# choose function if x is 1 choose y else choose z
def CH (x, y, z):
    # padding to 32-bit
    x = np.binary_repr(x)
    for i in range (32-len(x)):
        x = '0' + x
    y = np.binary_repr(y)
    for i in range (32-len(y)):
        y = '0' + y
    z = np.binary_repr(z)
    for i in range (32-len(z)):
        z = '0' + z
    output = 0

    for i in range (32):
        if x[31-i] == '1':
            if y[31-i] == '1':
                output += 2**i
        else:
            if z[31-i] == '1':
                output += 2**i

    return output

# get the more frequent data on same bit
def MAJ (x, y, z):
    # padding to 32-bit
    x = np.binary_repr(x)
    for i in range (32-len(x)):
        x = '0' + x
    y = np.binary_repr(y)
    for i in range (32-len(y)):
        y = '0' + y
    z = np.binary_repr(z)
    for i in range (32-len(z)):
        z = '0' + z
    output = 0

    for i in range (32):
        count1 = 0
        if x[31-i] == '1': count1 += 1
        if y[31-i] == '1': count1 += 1
        if z[31-i] == '1': count1 += 1 
        if count1 > 1: output += 2**i

    return output

# produce constants k and H0
k = np.ndarray (64, np.uint32)
H0 = np.ndarray (8, np.uint32)

count = 0
for n in range (2, 1000):
    isPrime = True
    for i in range (2, n):
        if n % i == 0: 
            isPrime = False
            break # not prime
    if isPrime:
        k[count] = n**(1/3) * 2**32
        if count < 8:
            H0[count] = n**(1/2) * 2**32
        count += 1
        if count == 64: break

# print k constant and H0 constant
'''
for i in range (64):
    #print (np.binary_repr(k[i]))
    print (k[i])
'''
'''
for i in range (8):
    #print (np.binary_repr(H0[i]))
    print (H0[i])
'''

# check if produced data is same as internet constant
#print (0x6a09e667 == H0[0])
#print (0xc67178f2 == k[63])

CH(100, 120, 150)
MAJ(100, 120, 150)



inputStr = "A hello world message to the programmers! Another hello world message to the programmers!"
l = len (inputStr)
print ('length of string:', l, 'characters')
print ('length of message:', l*8, 'bits')
block_size = 64

while l > block_size - 9:
    block_size *= 2

b = np.ndarray (block_size, np.uint8)
print ('block size:', block_size, end='\n\n')

# set last 64 bits as length size
if l*8 >= 2**8:
    b[1] = l*8 // 2**8
    b[0] = l*8 % 2**8
else:
    b[1] = 0
    b[0] = l * 8
# set message in block and padding
for i in range (block_size-2):
    if i < l:
        b[block_size-1-i] = ord(inputStr[i])
    elif i == l:
        b[block_size-1-i] = 2**7
    else:
        b[block_size-1-i] = 0


print ('message in binary:')
for i in range (block_size):
    print (np.binary_repr(b[block_size-1-i]), end='')
print ('\n')

print ('message in 8-bit integer:')
for i in range (block_size):
    print (b[block_size-1-i], end=' ')
    #print (b[i], end=' ')
print ('\n')

w = np.zeros ((block_size//64, 64), np.uint32)

# change from big endian to little endian

for i in range (block_size//64): # i is i-th block
    for j in range (16): # j is w's 0~15-th word
        sumOf32Bit = 0
        for block_i in range (4):
            sumOf32Bit += b[block_size-1-(4*j+block_i)-i*64] * 2**(24-block_i*8)
        w[i][j] = sumOf32Bit
    
    for t in range (16, 64):
        w[i][t] = SIGMA1 (w[i][t-2]) + w[i][t-7] + SIGMA0 (w[i][t-15]) + w[i][t-16]


# print message blocks
for i in range (block_size//64):
    print ('Block', i)
    for j in range (64):
        print (w[i][j], end='\t')
        if j % 8 == 7: print ()
    print ()

# compression
Hout = H0
Htmp = np.ndarray (8, np.uint32)

print ('H0\n', Hout, end='\n\n')
for i in range (block_size//64):
    print ('H', i+1, sep='')
    Htmp = Hout
    for t in range (64):
        T1 = SIGMA1(Htmp[4]) + CH(Htmp[4], Htmp[5], Htmp[6]) + Htmp[7] + k[t] + w[i][t]
        T2 = SIGMA0(Htmp[4]) + MAJ(Htmp[4], Htmp[5], Htmp[6])

        for j in range (6, 0, -1):
            Htmp[j+1] = Htmp[j]

        Htmp[0] = T1 + T2
        Htmp[4] += T1
    
    Hout = np.add(Hout, Htmp)
    print (Hout, end='\n\n')