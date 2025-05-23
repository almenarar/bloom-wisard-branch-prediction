from bloom_filter import BloomFilter


class Discriminator:
    def __init__(self, num_bloom_filters, lut_addr_size, num_hashes):
        self.bloom_filters = [
            BloomFilter(lut_addr_size, num_hashes) for _ in range(num_bloom_filters)
        ]
        self.num_bloom_filters = num_bloom_filters

    def train(self, input_pieces):
        for i in range(self.num_bloom_filters):
            self.bloom_filters[i].add_entry(input_pieces[i])

    def get_count(self, input_pieces):
        count = 0
        for i in range(self.num_bloom_filters):
            count += int(self.bloom_filters[i].check_entry(input_pieces[i]))
        return count

    def binarize(self, b):
        if b == 0:
            return
        for i in range(self.num_bloom_filters):
            self.bloom_filters[i].binarize(b)

    def forget(self, input_pieces):
        for i in range(self.num_bloom_filters):
            if self.bloom_filters[i].check_entry(input_pieces[i]):
                self.bloom_filters[i].remove_entry(input_pieces[i])