import pandas as pd
from pyfaidx import Fasta
import os
import time
import pickle

# === Setup paths ===
# script_path = os.path.abspath(__file__)
# base_dir = script_path.split("Contrastive_Learning")[0] + "Contrastive_Learning"
base_dir = "/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/"

split = "test"
csv_file_path = os.path.join(base_dir, f"{split}_cassette_exons.csv")
refseq_path = os.path.join("/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/refseq/hg38.fa")

output_path = os.path.join(base_dir)

os.makedirs(output_path, exist_ok=True)

# === Load data ===
df = pd.read_csv(csv_file_path)
# df = df[df['Retina - Eye'] != -1.0]  # Only exons with valid PSI values
short_exon = 0
filename = os.path.basename(csv_file_path)           # 'train_cassette_exons.csv'
prefix = filename.split('_')[0]     
# if prefix == 'variable':
#     df['chromosome'] = df['exon_location'].apply(lambda loc: loc.split(":")[0])


# === Helper functions ===
def reverse_complement(seq):
    return seq.translate(str.maketrans("ATCGatcg", "TAGCtagc"))[::-1]

def get_sequence(genome, chrom, start, end, strand):
    try:
        seq = genome[chrom][int(start)-1:int(end)].seq
        return reverse_complement(seq) if strand == '-' else seq
    except Exception as e:
        print(f"⚠️ Sequence error: {chrom}:{start}-{end} ({strand}) — {e}")
        return None

def calculate_5prime_intron(start, end, strand):
    if strand == '+':
        return (start - 300, start - 1) if start > 300 else (None, None)
    else:
        return end + 1, end + 300

def calculate_3prime_intron(start, end, strand):
    if strand == '+':
        return end + 1, end + 300
    else:
        return (start - 300, start - 1) if start > 300 else (None, None)

def get_exon_segments(genome, chrom, start, end, strand):
    global short_exon
    try:
        exon_len = end - start + 1
        if exon_len >= 200:
            e_start = genome[chrom][start-1:start-1+100].seq
            e_end   = genome[chrom][end-100:end].seq
        else:
            short_exon +=1
            half = exon_len // 2
            e_start = genome[chrom][start-1:start-1+half].seq
            e_end   = genome[chrom][start-1+half:end].seq
        if strand == '-':
            return reverse_complement(e_start), reverse_complement(e_end)
        return e_start, e_end
    except Exception as e:
        print(f"⚠️ Exon error: {chrom}:{start}-{end} ({strand}) — {e}")
        return None, None

# === Load genome ===
if not os.path.exists(refseq_path):
    refseq_path += '.gz'
genome = Fasta(refseq_path)

# === Storage ===
dict_5prime, dict_3prime, dict_exon, merged_dict = {}, {}, {}, {}

# === Process each exon ===
for _, row in df.iterrows():
    exon_name = row['exon_id']
    strand = row['exon_strand']
    chrom = row['chromosome']
    psi_val = row['pericyte']
    start = int(row['exon_location'].split(":")[1].split("-")[0])
    end = int(row['exon_location'].split(":")[1].split("-")[1])

    # 5′ intron
    i_start, i_end = calculate_5prime_intron(start, end, strand)
    if i_start and i_end:
        seq_5p = get_sequence(genome, chrom, i_start, i_end, strand)
        if seq_5p:
            dict_5prime[exon_name] = {'psi_val': psi_val, 'hg38': seq_5p}

    # 3′ intron
    i_start, i_end = calculate_3prime_intron(start, end, strand)
    if i_start and i_end:
        seq_3p = get_sequence(genome, chrom, i_start, i_end, strand)
        if seq_3p:
            dict_3prime[exon_name] = {'psi_val': psi_val, 'hg38': seq_3p}

    # Exon
    e_start, e_end = get_exon_segments(genome, chrom, start, end, strand)
    if e_start and e_end:
        dict_exon[exon_name] = {'psi_val': psi_val, 'hg38': {'start': e_start, 'end': e_end}}

    # Merge if all three components exist
    if seq_5p and seq_3p and e_start and e_end:
        merged_dict[exon_name] = {
            'psi_val': psi_val,
            '5p': seq_5p,
            '3p': seq_3p,
            'exon': {
                'start': e_start,
                'end': e_end
            }
        }

   
# === Save ===
with open(os.path.join(output_path, f'psi_{prefix}_pericyte_psi_5primeintron_sequences_dict.pkl'), 'wb') as f:
    pickle.dump(dict_5prime, f)
with open(os.path.join(output_path, f'psi_{prefix}_pericyte_psi_3primeintron_sequences_dict.pkl'), 'wb') as f:
    pickle.dump(dict_3prime, f)
with open(os.path.join(output_path, f'psi_{prefix}_pericyte_psi_exon_sequences_dict.pkl'), 'wb') as f:
    pickle.dump(dict_exon, f)
with open(os.path.join(output_path, f'psi_{prefix}_pericyte_psi_MERGED.pkl'), 'wb') as f:
    pickle.dump(merged_dict, f)
    
print("✅ All Tabula Sapiens exon sequences extracted and saved.")
print("short exons num ", short_exon)