import scanpy as sc

# Replace with your file name
filename = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025/HUMAN_SPLICING_FOUNDATION_Anndata_ATSE_counts_77212_junctions_30_waypoints_20250625_131216.h5ad"

# Read the AnnData object
adata = sc.read(filename)

# Check basic info
# print(adata)

# Searching Cell Ontology and Broad Class

print(adata.obs["cell_ontology_class"].unique())
# matches = adata.obs["old_index"].str.contains("TSP14_smartseq2_B134547_B20_D101532_B20_LI_proximal_Epithelial")
# print(matches)

# print(adata.obs["broad_cell_class"].unique())