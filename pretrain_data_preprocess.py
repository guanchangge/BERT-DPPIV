import os
import sys
input_file=sys.argv[1]
output_file=sys.argv[2]
kmer_v=int(sys.argv[3])
with open(input_file,'r') as f:
    with open(output_file,'w')as w:
        for line in f:
            if line[0]=='>':
                w.write('\n')
            else:
                seq=line.rstrip()
                kmer=[]
                for i in range(0,len(seq),kmer_v):
                    if i+kmer_v>len(seq):
                        
                        kmer.append(seq[i:len(seq)])
                    else:
                        
                        kmer.append(seq[i:i+kmer_v])
                w.write('%s\n'%(' '.join(kmer)))