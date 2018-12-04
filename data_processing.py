from Bio import SeqIO, Seq, Alphabet
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from itertools import product
import numpy as np
import re

k_size = 2

# Import protein letters and compute possible permutations of k-mers
chars = IUPAC.IUPACProtein.letters
kmer = [''.join(c) for c in product(chars, repeat=k_size)]
n = len(kmer)
#print('Nr. of possible k-mers:', n)
#print(kmer)

# Method for reading fasta file and return sequence
def read_sequence(filename):
        sequences = list()
        for seq_record in SeqIO.parse(open(filename, mode='r'), 'fasta'):
                seq_record.description=' '.join(seq_record.description.split()[1:])
                sequences.append(seq_record.seq)
        return sequences

# Read the sequences from 3 COG families:
COG1 = read_sequence("data/COG1.fasta")
COG160 = read_sequence("data/COG160.fasta")
COG161 = read_sequence("data/COG161.fasta")


# Counting k-mers in a sequence
def count_occurence(seq):
        count = np.zeros(n)
        for i in range(n):
            temp = 0
            temp += len(re.findall('(?='+kmer[i]+')', str(seq)))
            count[i] = temp
        return count

#for seq in family:        

def kmer_prob(family):
    kmer_matrix = np.zeros((len(family),n))
    index = 0
    for seq in family:
        kmer_matrix[index,:] = count_occurence(seq)
        index = index+1
    return kmer_matrix/len(family)

#Count how many times the k-mers appears in all families
COG1_prob = kmer_prob(COG1)
COG160_prob = kmer_prob(COG160)
COG161_prob = kmer_prob(COG161)

# Store the result in a npy file
np.save("data/COG1_prob.npy", COG1_prob)
np.save("data/COG160_prob.npy", COG160_prob)
np.save("data/COG161_prob.npy", COG161_prob)
