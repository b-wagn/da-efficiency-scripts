from dataclasses import dataclass

import math

@dataclass
class Code:
    size_msg_symbol: int  # size of one symbol in the message
    size_code_symbol: int # size of one symbol in the code
    msg_len: int          # number of symbols in the message
    codeword_len: int     # number of symbols in the codeword
    reception: int        # number of symbols needed to reconstruct (worse case)

    def interleave(self, ell):
        return Code(
            size_msg_symbol=self.size_msg_symbol * ell,
            size_code_symbol=self.size_code_symbol * ell,
            msg_len=self.msg_len,
            codeword_len=self.codeword_len,
            reception=self.reception
        )

    def tensor(self, col):
        assert self.size_msg_symbol == col.size_msg_symbol
        assert self.size_code_symbol == col.size_code_symbol
        assert self.size_msg_symbol == self.size_code_symbol

        row_dist = self.codeword_len - self.reception + 1
        col_dist = col.codeword_len - col.reception + 1
        codeword_len = self.codeword_len * col.codeword_len

        '''
        Example:

        D D | o o
        D D | o o
        ----+----
        o o | o o
        o o | o o

        Where D is the data.
        The reception is 8, since 7 is not enough to reconstruct:

        o o | o x
        o o | o x
        ----+----
        o o | o x
        x x | x x

        Given the symbols marked with x, I cannot reconstruct the data.
        '''

        return Code(
            size_msg_symbol=self.size_msg_symbol,
            msg_len=self.msg_len * col.msg_len,
            size_code_symbol=self.size_code_symbol,
            codeword_len=codeword_len,
            reception=codeword_len - row_dist * col_dist + 1
        )

    def __eq__(self, other):
        return (
            self.size_msg_symbol == other.size_msg_symbol
            and self.size_code_symbol == other.size_code_symbol
            and self.msg_len == other.msg_len
            and self.codeword_len == other.codeword_len
            and self.reception == other.reception
        )

    def is_identity(self):
        return (
            self.size_msg_symbol == self.size_code_symbol
            and self.msg_len == self.codeword_len
        )

    def relative_reception(self):
        return self.reception / self.codeword_len

    def num_samples_to_reconstruction(self, sec_par):
        # special case: if only one symbol is needed, we are done
        if self.reception == 1:
            return 1

        # special case: if all symbols are needed: just regular coupon collector
        if self.reception == self.codeword_len:
            n = self.codeword_len
            return (n / math.log(math.e, 2)) * (math.log(n, 2) + sec_par)
        
        # generalized coupon collector
        r = self.relative_reception()
        delta = self.reception - 1
        c = delta/self.codeword_len
        s = math.ceil(-sec_par / math.log2(c) + (1.0-math.log(math.e,c))*delta) 
        return int(s)

def makeTrivialCode(fsize, k):
    return Code(
        size_msg_symbol=fsize,
        msg_len=k,
        size_code_symbol=fsize,
        codeword_len=k,
        reception=k,
    )

# Reed-Solomon Code
# Polynomial of degree k-1 over field with field element length fsize
# Evaluated at n points
def makeRSCode(fsize, k, n):
    assert k <= n
    assert 2**fsize >= n, 'no such reed-solomon code :('
    return Code(
        size_msg_symbol=fsize,
        msg_len=k,
        size_code_symbol=fsize,
        codeword_len=n,
        reception=k
    )

# tests
assert makeRSCode(5, 2, 4).tensor(makeRSCode(5, 2, 4)).reception == 8
assert makeRSCode(5, 2, 4).reception == 2