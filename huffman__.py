"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    >>> byte_to_bits(32)
    '00100000'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Helper functions for HuffmanTree
def inter_lst(tree):
    """Return the list of internal nodes in postorder traversal.

     @param HuffmanNode tree: this HuffmanNode
     @rtype: list[HuffmanNode]

     >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
     >>> number_nodes(tree)
     >>> len(inter_lst(tree))
     1
     >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
     >>> right = HuffmanNode(5)
     >>> tree = HuffmanNode(None, left, right)
     >>> number_nodes(tree)
     >>> len(inter_lst(tree))
     2
     """
    lst = []
    if not tree.symbol:
        if tree.left.symbol and tree.right.symbol:
            lst.append(tree)
        else:
            if not tree.left.symbol:
                lst += inter_lst(tree.left)
            if not tree.right.symbol:
                lst += inter_lst(tree.right)
            lst.append(tree)
        return lst


def full_lst(tree):
    """Return the list of all nodes in postorder traversal.

    @param HuffmanNode tree: this HuffmanNode
    @rtype: list[HuffmanNode]

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> len(full_lst(tree))
    3
    """
    lst = []
    if not tree.left and not tree.right:
        lst.append(tree)
        return lst
    else:
        if tree.left:
            lst = lst + full_lst(tree.left)
        if tree.right:
            lst = lst + full_lst(tree.right)
        lst.append(tree)
        return lst


# ====================
# Functions for compression
def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    # todo
    d = {}
    for w in text:
        if w in d:
            d[w] += 1
        else:
            d[w] = 1
    return d


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    >>> freq = {1:12, 2:33, 3:15, 4:8, 5:32}
    >>> t = huffman_tree(freq)
    >>> t1 = HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(1), HuffmanNode(4)), HuffmanNode(3)), \
    HuffmanNode(None, HuffmanNode(5), HuffmanNode(2)))
    >>> t == t1
    True
    """
    # todo |start from easiest statement|
    lst = []
    for i in freq_dict:
        lst.append([freq_dict[i], HuffmanNode(i)])
    lst.sort()
    while len(lst) > 1:
        m = lst.pop(0)
        n = lst.pop(0)
        lst.append([m[0] + n[0], HuffmanNode(None, m[1], n[1])])
        lst.sort()
    return lst[0][1]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree = HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(1), HuffmanNode(4)), HuffmanNode(3)), \
    HuffmanNode(None, HuffmanNode(5), HuffmanNode(2)))
    >>> get_codes(tree)
    {1: "000", 2:"11", 3:"01", 4:"001", 5:"10"}
    """
    # todo **
    codes = {}

    def preorder(tree, bit_str):
        """
        Preorder traversal for huffman tree
        @param tree: HuffmanNode
        @param bit_str: str
        @rtype: dict
        """
        if tree.left:
            preorder(tree.left, bit_str + "0")
        if tree.right:
            preorder(tree.right, bit_str + "1")
        if tree.is_leaf():
            codes[tree.symbol] = bit_str

    preorder(tree, "")
    return codes


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # todo
    internal_lst = inter_lst(tree)
    i = 0
    while i in range(len(internal_lst)):
        item = internal_lst[i]
        item.number = i
        i += 1


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    # todo aveage = (frequency * length of bits) / (sum of frequencies)
    length_dict = get_codes(tree)
    sum_variable = 0
    sum_freq = sum([freq_dict[c] for c in freq_dict.keys()])
    for c in freq_dict.keys():
        sum_variable += int(freq_dict[c]) * len(length_dict[c])
    return sum_variable / sum_freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # todo
    byte = []
    string = ""
    for c in text:
        string += codes[c]
    string += "0" * (8 - len(string) % 8)
    num = int(len(string) / 8)
    for k in range(0, num):
        byte.append(bits_to_byte(string[k*8:(k+1)*8]))
    return bytes(byte)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    # todo
    result = bytes([])
    lst = full_lst(tree)[0:-1]
    for c in lst:
        if c.symbol:
            result += bytes([0, c.symbol])
        else:
            result += bytes([1, c.number])
    return result


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    # todo
    root_readnode = node_lst[root_index]
    node_lst.remove(root_readnode)
    if not root_readnode.l_type or not root_readnode.r_type:
        left = HuffmanNode(root_readnode.l_data)
        right = HuffmanNode(root_readnode.r_data)
        if not root_readnode.l_type and not root_readnode.r_type:
            return HuffmanNode(None, left, right)
        else:
            if root_readnode.l_type:
                left = generate_tree_general(node_lst, 0)
            if root_readnode.r_type:
                right = generate_tree_general(node_lst, 0)
            return HuffmanNode(None, left, right)
    else:
        right = generate_tree_general(node_lst, 0)
        left = generate_tree_general(node_lst, 0)
        if root_readnode.l_data > root_readnode.r_data:
            return HuffmanNode(None, left, right)
        else:
            return HuffmanNode(None, right, left)


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    # todo
    root_readnode = node_lst[root_index]
    node_lst.remove(root_readnode)
    if not root_readnode.l_type or not root_readnode.r_type:
        left = HuffmanNode(root_readnode.l_data)
        right = HuffmanNode(root_readnode.r_data)
        if not root_readnode.l_type and not root_readnode.r_type:
            return HuffmanNode(None, left, right)
        else:
            if root_readnode.left_type:
                left = generate_tree_general(node_lst, 0)
            if root_readnode.r_type:
                right = generate_tree_general(node_lst, 0)
            return HuffmanNode(None, left, right)
    else:
        left = generate_tree_general(node_lst, 0)
        right = generate_tree_general(node_lst, 0)
        return HuffmanNode(None, left, right)


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    # todo
    byte = []
    a = ""
    n = 0
    lst = [byte_to_bits(item) for item in text]
    for c in lst:
        a += c
    for c in range(0, size):
        while tree.left is not None and tree.right is not None:
            if a[n] == "0":
                tree = tree.left
            else:
                tree = tree.right
            n += 1
        byte += [tree.symbol]
    return bytes(byte)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # todo
    def check_value(t, value):
        if t.symbol:
            return t.symbol == value
        else:
            return check_value(t.left, value) or \
                   check_value(t.right, value)

    def height(t, value):
        if value == t.symbol:
            return 1
        elif check_value(t.left, value):
            return 1 + height(t.left, value)
        else:
            return 1 + height(t.right, value)

    def contained_tree(t, value):
        if t.symbol == value:
            return t
        elif t.left is not None and t.right is not None:
                if check_value(t.left, value):
                    return contained_tree(t.left, value)
                elif check_value(t.right, value):
                    return contained_tree(t.right, value)

    freq_height = {}
    for c in freq_dict:
        freq_height[c] = [freq_dict[c], height(tree, c)]
    for a in freq_height:
        for b in freq_height:
            if freq_height[a][0] >= freq_height[b][0] and \
                            freq_height[a][1] >= freq_height[b][1]:
                tree1 = contained_tree(tree, a)
                tree2 = contained_tree(tree, b)
                tree1.symbol, tree2.symbol = tree2.symbol, tree1.symbol


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    # TODO: Uncomment these when you have implemented all the functions
    # import doctest
    # doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
