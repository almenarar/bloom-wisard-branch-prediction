import mmh3

class BloomFilter:
    def __init__(self, lut_addr_size, num_hashes):
        self.lut = dict()
        self.lut_addr_size = lut_addr_size  # bits
        self.num_hashes = num_hashes
        self.seed = 0

    def add_entry(self, input_piece):
        for i in range(self.num_hashes):
            addr = mmh3.hash(input_piece, self.seed + i) % 2**self.lut_addr_size
            if addr in self.lut:
                self.lut[addr] += 1
            else:
                self.lut[addr] = 1

    def remove_entry(self, input_piece):
        for i in range(self.num_hashes):
            addr = mmh3.hash(input_piece, self.seed + i) % 2**self.lut_addr_size
            if addr in self.lut:
                self.lut[addr] -= 1
                if self.lut[addr] == 0:
                    del self.lut[addr]

    def check_entry(self, input_piece):
        for i in range(self.num_hashes):
            addr = mmh3.hash(input_piece, self.seed + i) % 2**self.lut_addr_size
            if addr not in self.lut:
                return False
        return True

    def binarize(self, b):
        for k in self.lut:
            if self.lut[k] >= b:
                self.lut[k] = 1
            else:
                self.lut[k] = 0
        for k in list(self.lut.keys()):
            if self.lut[k] == 0:
                del self.lut[k]