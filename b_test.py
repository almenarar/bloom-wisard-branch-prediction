import random

from bloom_filter import BloomFilter

def test_binarize_b(data, lut_addr_size, num_hashes, b_values):
    """
    Testa a função binarize com diferentes valores de b e mede a taxa de falsos positivos.

    Args:
        data: Lista de strings para adicionar ao filtro.
        lut_addr_size: Tamanho do filtro.
        num_hashes: Número de funções hash.
        b_values: Lista de valores para testar para binarize.

    Returns:
        Um dicionário contendo a taxa de falsos positivos para cada valor de b.
    """

    false_positive_rates = {}
    for b in b_values:
        bloom_filter = BloomFilter(lut_addr_size, num_hashes)
        for item in data:
            bloom_filter.add_entry(item)
        bloom_filter.binarize(b)

        # Gera alguns elementos de teste que NÃO estão no conjunto original
        test_data = [f"test_{i}" for i in range(1000)]
        false_positives = 0
        for item in test_data:
            if bloom_filter.check_entry(item):
                false_positives += 1
        false_positive_rate = false_positives / len(test_data)
        false_positive_rates[b] = false_positive_rate

    return false_positive_rates

if __name__ == '__main__':
    data = ["apple", "banana", "cherry", "apple", "banana", "apple", "date", "fig", "apple"]
    lut_addr_size = 6
    num_hashes = 3
    b_values = [1, 2, 3]  # Valores de b para testar

    results = test_binarize_b(data, lut_addr_size, num_hashes, b_values)
    print("Taxa de Falsos Positivos para Diferentes Valores de b:")
    for b, rate in results.items():
        print(f"b = {b}: Taxa de Falsos Positivos = {rate:.4f}")