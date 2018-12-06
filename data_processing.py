# Import modules
from Bio import SeqIO, Seq, Alphabet
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from itertools import product
import numpy as np
import re

# Status:
print("Starting preprocessing...")

# Import protein letters and compute possible permutations of k-mers
k_size = 2
chars = IUPAC.IUPACProtein.letters
kmer = [''.join(c) for c in product(chars, repeat=k_size)]
n = len(kmer)

# Method for reading fasta file and return sequence
def read_sequence(filename):
        sequences = list()
        for seq_record in SeqIO.parse(open(filename, mode='r'), 'fasta'):
                seq_record.description=' '.join(seq_record.description.split()[1:])
                sequences.append(seq_record.seq)
        return sequences

# Read the sequences from 3 COG families:
COG1 = read_sequence("fasta_files/COG1.fasta")
COG160 = read_sequence("fasta_files/COG160.fasta")
COG161 = read_sequence("fasta_files/COG161.fasta")


# Counting k-mers in a sequence
def count_occurence(seq):
        count = np.zeros(n)
        for i in range(n):
            temp = 0
            temp += len(re.findall('(?='+kmer[i]+')', str(seq)))
            count[i] = temp
        return count

#Start a counting for each sequence in a family:        
def kmer_count(family):
    kmer_matrix = np.zeros((len(family),n))
    index = 0
    for seq in family:
        kmer_matrix[index,:] = count_occurence(seq)/(len(seq)+2-k_size)
        index = index+1
    return kmer_matrix

#Count how many times the k-mers appears in all families
COG1_prob = kmer_count(COG1)
COG160_prob = kmer_count(COG160)
COG161_prob = kmer_count(COG161)

# Store the result in a npy file
np.save("preprocessed_data/COG1_prob.npy", COG1_prob)
np.save("preprocessed_data/COG160_prob.npy", COG160_prob)
np.save("preprocessed_data/COG161_prob.npy", COG161_prob)

# Status:
print("Preprocessing done!")