import pandas as pd

metadata_path = "/gpfs/commons/home/nkeung/tabula_muris_data/bam_paths.csv"

ts = pd.read_csv(metadata_path)

cell_counts = ts['cell_ontology_class'].value_counts().reset_index()
cell_counts.columns = ['cell_ontology_class', 'count']

filtered_cells = cell_counts[(cell_counts['count'] >= 30)]
filtered_cells = filtered_cells.sort_values(by='count', ascending=False)

filtered_cells.to_csv("/gpfs/commons/home/nkeung/tabula_muris_data/cell_counts.csv", index=False)
print(filtered_cells)
