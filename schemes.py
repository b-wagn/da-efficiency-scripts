#!/usr/bin/env python

from codes import *
from dataclasses import dataclass
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


@dataclass
class Scheme:
    code: Code            # code that is used
    com_size: int         # size of commitment in bits
    opening_overhead: int  # overhead of opening a symbol in the encoding

    def samples(self):
        '''
        i.e. the number of random samples needed to collect 
        enough symbols except with small probability
        '''
        return self.code.samples

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


def makeNaiveScheme(datasize):
    '''
    Naive scheme:
    Put all the data in one symbol, and let the commitment be a hash
    '''
    return Scheme(
        code=Code(
            size_msg_symbol=datasize,
            msg_len=1,
            size_code_symbol=datasize,
            codeword_len=1,
            reception=1,
            samples=1
        ),
        com_size=HASH_SIZE,
        opening_overhead=0
    )


def makeMerkleScheme(datasize, chunksize=1024):
    '''
    Merkle scheme:
    Take a merkle tree and the identity code
    Every leaf contains chunksize bits of the data
    '''
    k = math.ceil(datasize / chunksize)
    return Scheme(
        code=makeTrivialCode(chunksize, k),
        com_size=HASH_SIZE,
        opening_overhead=math.ceil(math.log(k, 2))*HASH_SIZE
    )


def makeKZGScheme(datasize, invrate=4):
    '''
    KZG Commitment, interpreted as an erasure code commitment for the RS code
    The RS Code is set to have parameters k,n with n = invrate * k
    '''
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


def makeTensorScheme(datasize, invrate=2):
    '''
    Tensor Code Commitment, where each dimension is expanded with inverse rate invrate.
    That is, data is a k x k matrix, and the codeword is a n x n matrix, with n = invrate * k
    Both column and row code are RS codes.
    '''
    m = math.ceil(datasize / BLS_FE_SIZE)
    k = math.ceil(math.sqrt(m))
    n = invrate * k

    rs = makeRSCode(BLS_FE_SIZE, k, n)

    return Scheme(
        code=rs.tensor(rs),
        com_size=BLS_GE_SIZE * k,
        opening_overhead=BLS_GE_SIZE,
    )


def makeHashBasedScheme(datasize, fsize=32, P=8, L=64, invrate=4):
    '''
    Hash-Based Code Commitment over field with elements of size fsize,
    parallel repetition parameters P and L. Data is treated as a k x k matrix,
    and codewords are k x n matrices, where n = k*invrate.
    '''
    m = math.ceil(datasize / fsize)
    k = math.ceil(math.sqrt(m))
    n = invrate * k
    rs = makeRSCode(fsize, k, n)

    return Scheme(
        code=rs.interleave(k),
        com_size=n * HASH_SIZE + P * n * fsize + L * k * fsize,
        opening_overhead=0,
    )


def makeHomHashBasedScheme(datasize, P=2, L=2, invrate=4):
    '''
    Homomorphic Hash-Based Code Commitment instantiated with Pedersen Hash and
    parallel repetition parameters P and L. Data is treated as a k x k matrix,
    and codewords are k x n matrices, where n = k*invrate.
    '''
    m = math.ceil(datasize / PEDERSEN_FE_SIZE)
    k = math.ceil(math.sqrt(m))
    n = invrate * k
    rs = makeRSCode(PEDERSEN_FE_SIZE, k, n)

    return Scheme(
        code=rs.interleave(k),
        com_size=n * PEDERSEN_GE_SIZE + P * n *
        PEDERSEN_FE_SIZE + L * k * PEDERSEN_FE_SIZE,
        opening_overhead=0,
    )
