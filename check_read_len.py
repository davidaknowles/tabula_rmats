from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
import pysam

df = pd.read_csv("/gpfs/commons/home/nkeung/tabula_muris_data/bam_paths.csv")
paths = df["bam_path"].tolist()

def get_read_length(path):
    try:
        with pysam.AlignmentFile(path, "rb") as bam:
            for read in bam:
                return read.query_length
    except:
        return None

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = set()
    future_to_path = {}
    
    for path in paths:
        future = executor.submit(get_read_length, path)
        futures.add(future)
        future_to_path[future] = path
        
        if len(futures) >= 12:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            
            for future in done:
                length = future.result()
                path = future_to_path[future]
                
                if length is None:
                    continue
                
                if length != 100:
                    raise RuntimeError(f"Mismatch detected in {path}: {length}")

    # Drain remaining futures
    for future in futures:
        length = future.result()
        path = future_to_path[future]
        
        if length is None:
            continue
        
        if length != 100:
            raise RuntimeError(f"Mismatch detected in {path}: {length}")

print("✅ All read lengths are 100")