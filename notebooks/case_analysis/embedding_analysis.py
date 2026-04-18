import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
import random
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def find_functional_groups_scalable(df, min_shared_terms=2, min_group_size=3, 
                                   max_groups_per_ontology=20, sample_size=10000):
    """
    Scalable approach to find functionally similar protein groups
    """
    functional_groups = {}
    
    for ontology in df['ontology'].unique():
        print(f"Processing {ontology} ontology...")
        ont_df = df[df['ontology'] == ontology].copy()
        
        # Sample if dataset is too large
        if len(ont_df) > sample_size:
            ont_df = ont_df.sample(n=sample_size, random_state=42)
        
        groups = find_groups_by_shared_terms(ont_df, min_shared_terms, min_group_size)
        
        # Sort groups by average performance and take top groups
        for group in groups:
            avg_fmax = np.mean([p['fmax'] for p in group])
            for p in group:
                p['group_avg_fmax'] = avg_fmax
        
        groups.sort(key=lambda g: g[0]['group_avg_fmax'], reverse=True)
        functional_groups[ontology] = groups[:max_groups_per_ontology]
        
        print(f"Found {len(functional_groups[ontology])} groups for {ontology}")
    
    return functional_groups
def find_groups_by_shared_terms(df, min_shared_terms=2, min_group_size=3):
    """
    Find groups by exact term matching - much faster than similarity matrix
    """
    # Create a mapping from GO terms to proteins
    term_to_proteins = defaultdict(list)
    
    for idx, row in df.iterrows():
        protein_id = row['proteins']
        annotations = row['annotations']
        
        for term in annotations:
            term_to_proteins[term].append({
                'protein': protein_id,
                'fmax': row['fmax'],
                'auprc': row['auprc'],
                'annotations': annotations
            })
    
    # Find combinations of terms that appear together frequently
    protein_to_terms = {}
    for idx, row in df.iterrows():
        protein_to_terms[row['proteins']] = set(row['annotations'])
    
    # Group proteins by shared term sets
    groups = []
    processed_proteins = set()
    
    # Sort terms by frequency (start with less common terms for more specific groups)
    term_frequencies = [(term, len(proteins)) for term, proteins in term_to_proteins.items()]
    term_frequencies.sort(key=lambda x: x[1])
    
    for term, freq in term_frequencies:
        if freq < min_group_size:  # Skip terms with too few proteins
            continue
            
        term_proteins = term_to_proteins[term]
        
        # Find proteins that share multiple terms with this set
        candidates = []
        for prot_info in term_proteins:
            if prot_info['protein'] in processed_proteins:
                continue
                
            prot_terms = protein_to_terms[prot_info['protein']]
            
            # Count shared terms with other proteins having this term
            shared_counts = []
            for other_prot_info in term_proteins:
                if other_prot_info['protein'] != prot_info['protein']:
                    other_terms = protein_to_terms[other_prot_info['protein']]
                    shared = len(prot_terms.intersection(other_terms))
                    if shared >= min_shared_terms:
                        shared_counts.append((other_prot_info, shared))
            
            if len(shared_counts) >= min_group_size - 1:  # -1 because we count the protein itself
                candidates.append((prot_info, shared_counts))
        
        # Form groups from candidates
        if len(candidates) >= min_group_size:
            # Create a group with highly connected proteins
            group = []
            for prot_info, shared_list in candidates[:min_group_size*2]:  # Limit group size
                if prot_info['protein'] not in processed_proteins:
                    group.append(prot_info)
                    processed_proteins.add(prot_info['protein'])
                    
                    if len(group) >= min_group_size and len(group) <= 15:  # Max group size
                        break
            
            if len(group) >= min_group_size:
                groups.append(group)
    
    return groups
def find_functional_groups_by_clustering(df, n_clusters=50, max_samples=20000):
    """
    Alternative approach using TF-IDF and clustering for very large datasets
    """
    functional_groups = {}
    
    for ontology in df['ontology'].unique():
        print(f"Clustering {ontology} ontology...")
        ont_df = df[df['ontology'] == ontology].copy()
        
        # Sample if too large
        if len(ont_df) > max_samples:
            ont_df = ont_df.sample(n=max_samples, random_state=42)
        
        # Convert annotations to text for TF-IDF
        annotation_texts = []
        protein_info = []
        
        for idx, row in ont_df.iterrows():
            annotation_text = ' '.join(row['annotations'])
            annotation_texts.append(annotation_text)
            protein_info.append({
                'protein': row['proteins'],
                'fmax': row['fmax'],
                'auprc': row['auprc'],
                'annotations': row['annotations']
            })
        
        if len(annotation_texts) < n_clusters:
            n_clusters = len(annotation_texts) // 3
        
        if n_clusters < 2:
            continue
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
        tfidf_matrix = vectorizer.fit_transform(annotation_texts)
        
        # Mini-batch K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group proteins by clusters
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(protein_info[i])
        
        # Filter clusters by size and quality
        quality_clusters = []
        for cluster_id, proteins in clusters.items():
            if len(proteins) >= 3:  # Minimum cluster size
                # Calculate cluster quality (average pairwise GO term overlap)
                quality_score = calculate_cluster_quality(proteins)
                if quality_score > 0.1:  # Minimum quality threshold
                    quality_clusters.append((proteins, quality_score))
        
        # Sort by quality and take top clusters
        quality_clusters.sort(key=lambda x: x[1], reverse=True)
        functional_groups[ontology] = [cluster[0] for cluster in quality_clusters[:20]]
        
        print(f"Found {len(functional_groups[ontology])} quality clusters for {ontology}")
    
    return functional_groups
def calculate_cluster_quality(proteins):
    """
    Calculate the quality of a cluster based on GO term overlap
    """
    if len(proteins) < 2:
        return 0
    
    total_similarity = 0
    comparisons = 0
    
    # Sample pairs if cluster is too large
    if len(proteins) > 20:
        protein_sample = random.sample(proteins, 20)
    else:
        protein_sample = proteins
    
    for i in range(len(protein_sample)):
        for j in range(i + 1, len(protein_sample)):
            terms1 = set(protein_sample[i]['annotations'])
            terms2 = set(protein_sample[j]['annotations'])
            
            if len(terms1.union(terms2)) > 0:
                jaccard = len(terms1.intersection(terms2)) / len(terms1.union(terms2))
                total_similarity += jaccard
                comparisons += 1
    
    return total_similarity / comparisons if comparisons > 0 else 0
def select_representative_cases_efficient(df, n_cases_per_category=100):
    """
    Efficiently select representative cases without computing full similarity matrices
    """
    selected_cases = {}
    
    for ontology in df['ontology'].unique():
        ont_df = df[df['ontology'] == ontology].copy()
        
        cases = {}
        
        # 1. Performance-based selection (unchanged)
        ont_df['combined_score'] = (ont_df['fmax'] + ont_df['auprc']) / 2
        cases['best_performers'] = ont_df.nlargest(n_cases_per_category, 'combined_score')['proteins'].tolist()
        cases['worst_performers'] = ont_df.nsmallest(n_cases_per_category, 'combined_score')['proteins'].tolist()
        
        # 2. Annotation-based selection
        ont_df['n_annotations'] = ont_df['annotations'].apply(len)
        cases['highly_annotated'] = ont_df.nlargest(n_cases_per_category, 'n_annotations')['proteins'].tolist()
        cases['sparsely_annotated'] = ont_df.nsmallest(n_cases_per_category, 'n_annotations')['proteins'].tolist()
        
        # 3. Rare vs common function selection
        all_terms = []
        for annotations in ont_df['annotations']:
            all_terms.extend(annotations)
        
        term_counts = Counter(all_terms)
        common_terms = set([term for term, count in term_counts.most_common(100)])  # Top 100 most common
        rare_terms = set([term for term, count in term_counts.items() if count <= 5])  # Very rare terms
        
        # Proteins with mostly common functions
        ont_df['common_term_ratio'] = ont_df['annotations'].apply(
            lambda x: len([t for t in x if t in common_terms]) / max(len(x), 1)
        )
        cases['common_functions'] = ont_df.nlargest(n_cases_per_category, 'common_term_ratio')['proteins'].tolist()
        
        # Proteins with rare functions
        ont_df['rare_term_count'] = ont_df['annotations'].apply(
            lambda x: len([t for t in x if t in rare_terms])
        )
        cases['rare_functions'] = ont_df.nlargest(n_cases_per_category, 'rare_term_count')['proteins'].tolist()
        
        selected_cases[ontology] = cases
    
    return selected_cases
def comprehensive_case_selection_scalable(df, esm_embeddings, model_embeddings, 
                                        max_sample_size=20000):
    """
    Scalable comprehensive case selection for large datasets
    """
    print("Starting scalable case selection...")
    
    # Sample the dataset if it's too large for some analyses
    if len(df) > max_sample_size:
        print(f"Sampling {max_sample_size} proteins from {len(df)} total proteins...")
        df_sample = df.sample(n=max_sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    # 1. Efficient performance-based selection
    print("Selecting performance-based cases...")
    performance_cases = select_representative_cases_efficient(df)
    
    # 2. Functional groups using scalable methods
    print("Finding functional groups...")
    functional_groups = find_functional_groups_scalable(df_sample)
    
    # 3. Clustering-based groups (alternative approach)
    print("Finding clustering-based groups...")
    clustering_groups = find_functional_groups_by_clustering(df_sample)
    
    # 4. Embedding change analysis (only on available embeddings)
    print("Analyzing embedding changes...")
    available_proteins = set(esm_embeddings.keys()).intersection(set(model_embeddings.keys()))
    df_embeddings = df[df['proteins'].isin(available_proteins)].copy()
    
    if len(df_embeddings) > 10000:  # Sample for embedding analysis
        df_embeddings = df_embeddings.sample(n=10000, random_state=42)
    
    embedding_cases = analyze_embedding_changes_efficient(df_embeddings, esm_embeddings, model_embeddings)
    
    return {
        'performance_based': performance_cases,
        'functional_groups': functional_groups,
        'clustering_groups': clustering_groups,
        'embedding_changes': embedding_cases
    }
def analyze_embedding_changes_efficient(df, esm_embeddings, model_embeddings, 
                                      dimension_reduction_method='pca'):
    """
    Efficient embedding change analysis with different embedding dimensions
    ESM: 1280 dimensions, Model: 128 dimensions
    """
    embedding_changes = {}
    
    # First, collect all ESM embeddings to fit PCA
    print("Preparing embedding dimension alignment...")
    
    available_proteins = []
    esm_emb_list = []
    model_emb_list = []
    
    for idx, row in df.iterrows():
        protein_id = row['proteins']
        if protein_id in esm_embeddings and protein_id in model_embeddings:
            available_proteins.append(protein_id)
            esm_emb_list.append(esm_embeddings[protein_id])
            model_emb_list.append(model_embeddings[protein_id])
    
    if len(available_proteins) == 0:
        print("No proteins found with both ESM and model embeddings!")
        return {}
    
    esm_emb_array = np.array(esm_emb_list)  # Shape: (n_proteins, 1280)
    model_emb_array = np.array(model_emb_list)  # Shape: (n_proteins, 128)
    
    print(f"ESM embeddings shape: {esm_emb_array.shape}")
    print(f"Model embeddings shape: {model_emb_array.shape}")
    
    # Reduce ESM embeddings to 128 dimensions
    if dimension_reduction_method == 'pca':
        print("Applying PCA to reduce ESM embeddings from 1280 to 128 dimensions...")
        
        # Standardize ESM embeddings
        scaler = StandardScaler()
        esm_scaled = scaler.fit_transform(esm_emb_array)
        
        # Apply PCA
        pca = PCA(n_components=128, random_state=42)
        esm_reduced = pca.fit_transform(esm_scaled)
        
        print(f"PCA explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
    elif dimension_reduction_method == 'truncate':
        print("Truncating ESM embeddings to first 128 dimensions...")
        esm_reduced = esm_emb_array[:, :128]
        
    elif dimension_reduction_method == 'random_projection':
        print("Using random projection to reduce ESM embeddings...")
        from sklearn.random_projection import GaussianRandomProjection
        
        rp = GaussianRandomProjection(n_components=128, random_state=42)
        esm_reduced = rp.fit_transform(esm_emb_array)
        
    else:
        raise ValueError(f"Unknown dimension reduction method: {dimension_reduction_method}")
    
    print(f"Reduced ESM embeddings shape: {esm_reduced.shape}")
    
    # Now calculate embedding changes for each ontology
    for ontology in df['ontology'].unique():
        print(f"Processing {ontology} ontology...")
        ont_df = df[df['ontology'] == ontology].copy()
        
        changes = []
        for idx, row in ont_df.iterrows():
            protein_id = row['proteins']
            
            if protein_id in esm_embeddings and protein_id in model_embeddings:
                # Get the index of this protein in our arrays
                try:
                    prot_idx = available_proteins.index(protein_id)
                except ValueError:
                    continue
                
                esm_emb_reduced = esm_reduced[prot_idx]
                model_emb = model_emb_array[prot_idx]
                
                # Normalize embeddings
                esm_norm = esm_emb_reduced / (np.linalg.norm(esm_emb_reduced) + 1e-8)
                model_norm = model_emb / (np.linalg.norm(model_emb) + 1e-8)
                
                # Calculate cosine similarity and distance
                cosine_sim = np.dot(esm_norm, model_norm)
                cosine_distance = 1 - cosine_sim
                
                # Calculate L2 distance
                l2_distance = np.linalg.norm(esm_norm - model_norm)
                
                # Calculate component-wise correlation
                correlation = np.corrcoef(esm_emb_reduced, model_emb)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                changes.append({
                    'protein': protein_id,
                    'cosine_distance': cosine_distance,
                    'cosine_similarity': cosine_sim,
                    'l2_distance': l2_distance,
                    'correlation': correlation,
                    'fmax': row['fmax'],
                    'auprc': row['auprc'],
                    'annotations': row['annotations']
                })
        
        if len(changes) == 0:
            continue
        
        # Sort by different criteria
        embedding_changes[ontology] = {
            'largest_cosine_distance': sorted(changes, key=lambda x: x['cosine_distance'], reverse=True)[:50],
            'smallest_cosine_distance': sorted(changes, key=lambda x: x['cosine_distance'])[:50],
            'largest_l2_distance': sorted(changes, key=lambda x: x['l2_distance'], reverse=True)[:50],
            'smallest_l2_distance': sorted(changes, key=lambda x: x['l2_distance'])[:50],
            'highest_correlation': sorted(changes, key=lambda x: x['correlation'], reverse=True)[:50],
            'lowest_correlation': sorted(changes, key=lambda x: x['correlation'])[:50],
            'all_changes': changes  # Keep all for further analysis
        }
        
        print(f"  Found {len(changes)} proteins with embedding changes")
        print(f"  Mean cosine distance: {np.mean([c['cosine_distance'] for c in changes]):.4f}")
        print(f"  Mean L2 distance: {np.mean([c['l2_distance'] for c in changes]):.4f}")
        print(f"  Mean correlation: {np.mean([c['correlation'] for c in changes]):.4f}")
    
    return embedding_changes
