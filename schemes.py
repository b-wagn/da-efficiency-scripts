#!/usr/bin/env python

import math

# Some constants.
# Sizes of group elements, field elements, and hashes in bits
BLS_FE_SIZE = 48.0 * 8.0
BLS_GE_SIZE = 48.0 * 8.0

# Let's say we use the SECP256_k1 curve
PEDERSEN_FE_SIZE = 32.0 * 8.0
PEDERSEN_GE_SIZE = 33.0 * 8.0

# Let's say we use SHA256
HASH_SIZE = 256

# Statistical Security Parameter for Soundness
SECPAR_SOUND = 40

from dataclasses import dataclass

from codes import *

@dataclass
class Scheme:
    code: Code            # code that is used
    com_size: int         # size of commitment in bits
    opening_overhead: int # overhead of opening a symbol in the encoding

    def samples(self, sec_par=SECPAR_SOUND):
        '''
        i.e. the number of random samples needed to collect self.code.reception symbols
        except with probability 2^{-SECPAR_SOUND}.
        '''
        return self.code.num_samples_to_reconstruction(sec_par)

    def total_comm(self):
        '''
        Compute the total communication in bits.
        '''
        return self.comm_per_query() * self.samples()

    def comm_per_query(self):
        '''
        Compute the communication per query in bits.
        '''
        return math.log2(self.code.codeword_len) + self.opening_overhead + self.code.size_code_symbol

    def encoding_size(self):
        '''
        Compute the size of the encoding in bits.
        '''
        return self.code.codeword_len * (self.opening_overhead + self.code.size_code_symbol)

    def reception(self):
        '''
        Compute the reception of the code.
        '''
        return self.code.reception

    def encoding_length(self):
        '''
        Compute the length of the encoding.
        '''
        return self.code.codeword_len

# Naive scheme
# Put all the data in one symbol, and let the commitment be a hash
def makeNaiveScheme(datasize):
    return Scheme(
        code=Code(
            size_msg_symbol=datasize,
            msg_len=1,
            size_code_symbol=datasize,
            codeword_len=1,
            reception=1
        ),
        com_size=HASH_SIZE,
        opening_overhead=0
    )

# Merkle scheme
# Take a merkle tree and the identity code
def makeMerkleScheme(datasize, chunksize=1024):
    k = math.ceil(datasize / chunksize)
    return Scheme(
        code=Code(
            msg_len=k,
            codeword_len=k,
            reception=k,
            size_msg_symbol=chunksize,
            size_code_symbol=chunksize,
        ),
        com_size=HASH_SIZE,
        opening_overhead=math.ceil(math.log(k, 2))*HASH_SIZE
    )


# KZG Commitment, interpreted as an erasure code commitment for the RS code
# The RS Code is set to have parameters k,n with n = invrate * k
def makeKZGScheme(datasize, invrate=4):
    k = math.ceil(datasize / BLS_FE_SIZE)
    return Scheme(
        code=makeRSCode(
            BLS_FE_SIZE, 
            k, 
            k * invrate
        ),
        com_size=BLS_GE_SIZE,
        opening_overhead=BLS_GE_SIZE,
    )


# Tensor Code Commitment, where each dimension is expanded with inverse rate invrate.
# That is, data is a k x k matrix, and the codeword is a n x n matrix, with n = invrate * k
# Both column and row code are RS codes.
def makeTensorScheme(datasize, invrate=2):
    m = math.ceil(datasize / BLS_FE_SIZE)
    k = math.ceil(math.sqrt(m))
    n = invrate * k

    rs = makeRSCode(BLS_FE_SIZE, k, n)

    return Scheme(
        code=rs.tensor(rs),
        com_size=BLS_GE_SIZE*k,
        opening_overhead=BLS_GE_SIZE,
    )


# Hash-Based Code Commitment, over field with elements of size fsize,
# parallel repetition parameters P and L. Data is treated as a k x k matrix,
# and codewords are k x n matrices, where n = k*invrate.
def makeHashBasedScheme(datasize, fsize=32, P=8, L=64, invrate=4):
    m = math.ceil(datasize / fsize)
    k = math.ceil(math.sqrt(m))
    n = invrate * k
    rs = makeRSCode(fsize, k, n)

    return Scheme(
        code=rs.interleave(k),
        com_size=n * HASH_SIZE + P * n * fsize + L * k * fsize,
        opening_overhead=0,
    )

# Homomorphic Hash-Based Code Commitment
# instantiated with Pedersen Hash
# parallel repetition parameters P and L. Data is treated as a k x k matrix,
# and codewords are k x n matrices, where n = k*invrate.
def makeHomHashBasedScheme(datasize, P=2, L=2, invrate=4):
    m = math.ceil(datasize / PEDERSEN_FE_SIZE)
    k = math.ceil(math.sqrt(m))
    n = invrate * k
    rs = makeRSCode(PEDERSEN_FE_SIZE, k, n)

    return Scheme(
        code=rs.interleave(k),
        com_size=n * PEDERSEN_GE_SIZE + P * n * PEDERSEN_FE_SIZE + L * k * PEDERSEN_FE_SIZE,
        opening_overhead=0,
    )