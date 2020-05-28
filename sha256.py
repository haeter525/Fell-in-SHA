import warnings
import math
import numpy as np

np.set_printoptions(formatter={'int_kind':lambda x: '{:0>8X}'.format(x)}, linewidth=74)

BLOCK_SIZE = 64 # 64 bytes

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
def initial_constant() -> "ks":
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311]
    k = np.ndarray(64, np.uint32)

    for i in range(64):
        k[i] = 0
        value = PRIMES[i]**(1.0 / 3)
        value = (value - math.floor(value)) * 16
        for _ in range(8):
            m = math.floor(value)
            k[i] = (k[i] << 4) | (m & 0xF)
            value = (value - m) * 16

    return k
        
def initial_vector() -> "h0":
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]
    h = np.ndarray(8, np.uint32)

    for i in range(8):
        h[i] = 0
        value = math.sqrt(PRIMES[i])
        value = (value - math.floor(value)) * 16
        for _ in range(8):
            m = math.floor(value)
            h[i] = (h[i] << 4) | (m & 0xF)
            value = (value - m) * 16
    return h

def padding(byteArr) -> "padded":
    assert len(byteArr)<(2<<64), "Message is too large."

    l = len(byteArr)
    block_count = (l + 9) // BLOCK_SIZE + 1

    padded = np.ndarray (block_count * BLOCK_SIZE, np.uint8)
    #print ('block size:', BLOCK_SIZE, end='\n\n')


    # set message in block and padding
    for i in range (l): padded[i] = byteArr[i]
    padded[l] = 0x80
    for i in range (l+1, padded.shape[0]): padded[i] = 0

    # set last 64 bits as length size
    value = l*8
    #print('value=',value)
    for offsetByte in range(8):
        index = -(1 + offsetByte)
        padded[index] = value & 0xFF
        value >>= 8

    return padded

def message_as_block(padded) -> "blocks":
    assert (padded.shape[0] & 0x3F) == 0, "Length is not times of 64 bytes."

    block_count = padded.shape[0] // BLOCK_SIZE

    blocks = np.ndarray((block_count, BLOCK_SIZE//4), np.uint32)
    for c in range(blocks.shape[0]): # each message block
        for i in range(blocks.shape[1]): # each 32 bits
            index = c*BLOCK_SIZE + i*4
            blocks[c][i] = padded[index] << 24 | padded[index+1] << 16 | padded[index+2] << 8 | padded[index+3]

    return blocks

def message_schedule(block) -> "words":
    words = block.copy()
    words.resize(64)

    for i in range(15,64):
        words[i] = sigma1(words[i-2]) + words[i-7] + sigma0(words[i-15]) + words[i-16]
    return words

def message_compression(h0, words, ks) -> "hout":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    h = h0
    for i in range(64):
        temp1 = h[7]
        + CH(h[4], h[5], h[6])
        + SIGMA1(h[4])
        + words[i]
        + ks[i]

        temp2 = SIGMA0(h[0]) + MAJ(h[0], h[1], h[2])

        h[7] = h[6]
        h[6] = h[5]
        h[5] = h[4]
        h[4] = h[3] + temp1
        h[3] = h[2]
        h[2] = h[1]
        h[1] = h[0]
        h[0] = temp1 + temp2

        print('t{:d}'.format(i), h, sep='\n')

    warnings.filterwarnings("default", category=RuntimeWarning)
    return h
    
if __name__ == '__main__':
    inputStr = "abc"

    #Print basic info
    print ('string:', inputStr)
    byteArr = inputStr.encode('UTF-8')
    l = len(byteArr)
    print ('length of string:', l, 'characters')
    print ('length of message:', l*8, 'bits')

    #Padding
    padded = padding(byteArr)

    #As Blocks
    blocks = message_as_block(padded)

    print ('message in binary:')
    for block in blocks:
        for u32 in block:
            print (np.binary_repr(u32, 32), end=' ')
        print ()
    print ('\n')

    #Inital h0
    h = initial_vector()

    #Inital Constant
    ks = initial_constant()

    print('H0', h, sep='\n')

    for block_index in range( blocks.shape[0] ):
        block = blocks[block_index]

        #Message schedule
        words = message_schedule(block)

        #Message compression
        hadd = message_compression(h, words, ks)

        h = np.add(h, hadd)

        print('H{:d}'.format(block_index), h, sep='\n')

    #Print out result
    print()
    print('SHA256: ', end='')
    for part in h:
        print("{:0>8X}".format(part), sep='', end='')
    print()