# --- SPRA: SEMANTIC PRECURSOR RISK ASSESSMENT (FINAL COMPLETE v2) ---
# Target: International Journal of Computational Intelligence Systems
# Features: Logic Intact + Ablation Study Table + Model Saving + Full Benchmarks
# Status: FIXED (Array Conversion Bug Resolved)

import pandas as pd
import numpy as np
import torch
import re
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, brier_score_loss, f1_score, 
                             accuracy_score, precision_recall_curve, 
                             average_precision_score, auc, confusion_matrix, 
                             classification_report, precision_score, recall_score)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import pairwise_distances
import wikipedia

# Publication Settings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# =============================================================================
# 1. SPRA SCIENTIFIC PRIMITIVES (STRICTLY UNTOUCHED)
# =============================================================================

class PolysemyInstability:
    def __init__(self, encoder):
        self.encoder = encoder
        self.templates = ["{}", "Define {}", "Describe {} scientifically", "What implies {}?", "Explain {}"]
    def compute(self, q):
        try:
            vars = [t.format(q) for t in self.templates]
            embs = self.encoder.encode(vars)
            dists = pairwise_distances(embs, metric='cosine')
            return np.array([np.mean(dists[np.triu_indices_from(dists, k=1)])])
        except: return np.array([0.0])

class ManifoldIncoherence:
    def __init__(self, encoder): self.encoder = encoder
    def compute(self, q, c):
        try:
            sents = [s.strip() for s in re.split(r'[.!?]+', c) if len(s.split()) > 3]
            if len(sents) < 2: return np.array([0.0]) 
            s_embs = self.encoder.encode(sents)
            dists = pairwise_distances(s_embs, metric='cosine')
            return np.array([np.mean(dists[np.triu_indices_from(dists, k=1)])])
        except: return np.array([0.0])

class RelationalIsomorphism:
    def __init__(self, encoder):
        self.encoder = encoder
        self.rels = {'by', 'from', 'with', 'acquired', 'leads', 'owned', 'killed', 'invented', 'is', 'are', 'means', 'of', 'in', 'used', 'for', 'the'}
    def compute(self, q, c):
        try:
            def get_window_triples(text):
                text = re.sub(r'[^\w\s]', '', text) 
                tokens = text.lower().split()
                if len(tokens) < 3: return []
                triples = []
                for i in range(1, len(tokens)-1):
                    if tokens[i] in self.rels:
                        triples.append(f"{tokens[i-1]} {tokens[i]} {tokens[i+1]}")
                return triples
            q_tri = get_window_triples(q)
            c_tri = get_window_triples(c)
            if q_tri and not c_tri: return np.array([1.0])
            if q_tri and c_tri:
                q_emb = self.encoder.encode(q_tri).mean(axis=0)
                c_emb = self.encoder.encode(c_tri).mean(axis=0)
                sim = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-9)
                return np.array([1.0 - sim])
            return np.array([0.0])
        except: return np.array([0.0])

class AnchorDivergence:
    def __init__(self, encoder):
        self.encoder = encoder
        self.cache = {}
    def compute(self, q, c):
        triggers = ['what', 'define', 'meaning', 'who', 'used for', 'purpose']
        if not any(x in q.lower() for x in triggers): return np.array([0.5]) 
        try:
            subj = q.lower().split()[1] if len(q.split()) > 1 else ""
            for stop in ['is', 'are', 'the', 'a', 'an', 'mean', 'used', 'for']: subj = subj.replace(stop, "")
            if subj not in self.cache:
                try: self.cache[subj] = wikipedia.summary(subj, sentences=2)
                except: self.cache[subj] = ""
            anchor = self.cache[subj]
            if not anchor: return np.array([0.5])
            c_emb = self.encoder.encode([c])[0]
            a_emb = self.encoder.encode([anchor])[0]
            return np.array([1.0 - np.dot(c_emb, a_emb)])
        except: return np.array([0.5])

# =============================================================================
# 2. BASELINE MODELS
# =============================================================================

class Baseline_NLI:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    def compute_risk(self, queries, contexts):
        pairs = list(zip(queries, contexts))
        scores = self.model.predict(pairs) 
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()
        entailment_probs = probs[:, 1] 
        return 1.0 - entailment_probs

class Baseline_Cosine:
    def __init__(self, encoder):
        self.encoder = encoder
    def compute_risk(self, queries, contexts):
        q_embs = self.encoder.encode(queries, convert_to_tensor=True)
        c_embs = self.encoder.encode(contexts, convert_to_tensor=True)
        sims = torch.cosine_similarity(q_embs, c_embs).cpu().numpy()
        return (1.0 - sims) / 2.0 

# =============================================================================
# 3. SPRA SYSTEM (LOGIC INTACT)
# =============================================================================

class SPRA_Framework:
    def __init__(self):
        print("Initializing SPRA Framework...")
        self.enc = SentenceTransformer('all-MiniLM-L6-v2')
        self.polysemy = PolysemyInstability(self.enc)
        self.incoherence = ManifoldIncoherence(self.enc)
        self.isomorphism = RelationalIsomorphism(self.enc)
        self.anchor = AnchorDivergence(self.enc)
        self.clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        self.feat_names = ['Polysemy (SSA)', 'Incoherence (TSD)', 'Relational (RCG)', 'Anchor (PFV)', 'CosineDiv']

    def extract_features(self, df):
        X = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Signals"):
            q, c = row['question'], row['context']
            vec = []
            vec.extend(self.polysemy.compute(q))      
            vec.extend(self.incoherence.compute(q, c)) 
            vec.extend(self.isomorphism.compute(q, c)) 
            vec.extend(self.anchor.compute(q, c))      
            q_emb = self.enc.encode([q])[0]
            c_emb = self.enc.encode([c])[0]
            vec.append(1.0 - np.dot(q_emb, c_emb))     
            X.append(vec)
        return np.array(X)

    def predict_hybrid(self, q, c):
        f = []
        f.extend(self.polysemy.compute(q))
        f.extend(self.incoherence.compute(q, c))
        f.extend(self.isomorphism.compute(q, c))
        f.extend(self.anchor.compute(q, c))
        q_emb = self.enc.encode([q])[0]
        c_emb = self.enc.encode([c])[0]
        cosine_div = 1.0 - np.dot(q_emb, c_emb)
        f.append(cosine_div)
        feats = np.array(f)

        fact_threshold = 0.65 if cosine_div < 0.25 else 0.45
        if f[3] > fact_threshold: return 0.95, feats, "External Fact Mismatch (Wiki Divergence)"
        if f[1] > 0.4: return 0.95, feats, "Context Incoherence"
        if f[0] > 0.25: return 0.90, feats, "Query Ambiguity (Polysemy)"
        
        is_definition = any(x in q.lower() for x in ['define', 'meaning', 'explain', 'what is'])
        if is_definition:
            if cosine_div < 0.5: return 0.10, feats, "Definition Validated (Safe)"
            elif cosine_div > 0.6: return 0.90, feats, "Definition Topic Mismatch"
        if not is_definition and f[2] > 0.35: return 0.95, feats, "Relational Entity Mismatch"
        if f[2] < 0.25 and f[4] < 0.35: return 0.10, feats, "Structurally Validated (Safe)"

        prob = self.clf.predict_proba(feats.reshape(1, -1))[0, 1]
        return prob, feats, "Model Prediction"

# =============================================================================
# 4. BENCHMARKING ENGINE
# =============================================================================

def run_benchmark(df):
    spra = SPRA_Framework()
    bl_nli = Baseline_NLI()
    bl_cos = Baseline_Cosine(spra.enc)
    
    # Latency
    print("\n[Benchmark] Measuring Efficiency...")
    t0 = time.time()
    X = spra.extract_features(df)
    t_spra = (time.time() - t0) / len(df) * 1000
    
    t0 = time.time()
    _ = bl_cos.compute_risk(df['question'].tolist(), df['context'].tolist())
    t_cos = (time.time() - t0) / len(df) * 1000
    
    sample_df = df.sample(min(50, len(df)))
    t0 = time.time()
    _ = bl_nli.compute_risk(sample_df['question'].tolist(), sample_df['context'].tolist())
    t_nli = (time.time() - t0) / len(sample_df) * 1000

    y = df['is_hallucination'].astype(int).values
    
    # 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = {'SPRA': [], 'NLI': [], 'Cosine': []}
    y_true_all, spra_probs_all, nli_probs_all, cos_probs_all = [], [], [], []

    print("\n[Benchmark] Running 5-Fold Evaluation...")
    
    for tr, te in skf.split(X, y):
        # SPRA
        spra.clf.fit(X[tr], y[tr])
        p_spra = spra.clf.predict_proba(X[te])[:, 1]
        fold_aucs['SPRA'].append(roc_auc_score(y[te], p_spra))
        
        # Baselines
        q_te = df.iloc[te]['question'].tolist()
        c_te = df.iloc[te]['context'].tolist()
        p_nli = bl_nli.compute_risk(q_te, c_te)
        fold_aucs['NLI'].append(roc_auc_score(y[te], p_nli))
        p_cos = bl_cos.compute_risk(q_te, c_te)
        fold_aucs['Cosine'].append(roc_auc_score(y[te], p_cos))
        
        y_true_all.extend(y[te])
        spra_probs_all.extend(p_spra)
        nli_probs_all.extend(p_nli)
        cos_probs_all.extend(p_cos)

    # Stats
    _, p_nli = ttest_rel(fold_aucs['SPRA'], fold_aucs['NLI'])
    _, p_cos = ttest_rel(fold_aucs['SPRA'], fold_aucs['Cosine'])
    
    # --- FIX START: Convert lists to numpy arrays BEFORE comparison ---
    y_true_all = np.array(y_true_all)
    spra_probs_all = np.array(spra_probs_all)
    nli_probs_all = np.array(nli_probs_all)
    cos_probs_all = np.array(cos_probs_all)
    # --- FIX END ---

    # Tables
    print("\n=== TABLE 1: COMPARATIVE PERFORMANCE ===")
    print(f"{'Model':<15} | {'AUC':<8} | {'AP':<8} | {'Brier':<8} | {'p-value':<10}")
    print("-" * 60)
    for name, probs, pval in [('SPRA', spra_probs_all, '-'), ('NLI', nli_probs_all, f"{p_nli:.3f}"), ('Cosine', cos_probs_all, f"{p_cos:.3f}")]:
        auc_s = roc_auc_score(y_true_all, probs)
        ap_s = average_precision_score(y_true_all, probs)
        br_s = brier_score_loss(y_true_all, probs)
        print(f"{name:<15} | {auc_s:.4f}   | {ap_s:.4f}   | {br_s:.4f}   | {pval:<10}")

    print("\n=== TABLE 2: CLASSIFICATION METRICS (SPRA @ Threshold=0.5) ===")
    spra_preds = (spra_probs_all > 0.5).astype(int) # Now works because spra_probs_all is an array
    print(f"{'Accuracy':<15} | {accuracy_score(y_true_all, spra_preds):.4f}")
    print(f"{'Precision':<15} | {precision_score(y_true_all, spra_preds):.4f}")
    print(f"{'Recall':<15} | {recall_score(y_true_all, spra_preds):.4f}")
    print(f"{'F1-Score':<15} | {f1_score(y_true_all, spra_preds):.4f}")
    
    print("\n=== TABLE 3: EFFICIENCY ANALYSIS ===")
    print(f"{'SPRA':<15} | {t_spra:.2f} ms")
    print(f"{'NLI':<15} | {t_nli:.2f} ms")
    print(f"{'Cosine':<15} | {t_cos:.2f} ms")

    # --- NEW: ABLATION STUDY TABLE ---
    print("\n=== TABLE 4: ABLATION STUDY (Signal Importance) ===")
    print("Evaluating feature contributions by removing one signal at a time...")
    full_auc = roc_auc_score(y_true_all, spra_probs_all)
    feature_map = {'Polysemy': 0, 'Incoherence': 1, 'Relational': 2, 'Anchor': 3}
    print(f"{'Removed Signal':<20} | {'AUC Drop':<10}")
    print("-" * 35)
    
    drops = {}
    for name, idx in feature_map.items():
        # Drop the feature index
        mask = [i for i in range(X.shape[1]) if i != idx]
        # Train temporary classifier
        clf_abl = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_abl.fit(X[:, mask], y)
        p_abl = clf_abl.predict_proba(X[:, mask])[:, 1]
        drop = full_auc - roc_auc_score(y, p_abl)
        drops[name] = drop
        print(f"{name:<20} | -{drop:.4f}")

    # --- SAVE MODEL ---
    spra.clf.fit(X, y)
    model_path = os.path.join(os.getcwd(), 'spra_model.pkl')
    joblib.dump(spra.clf, model_path)
    print(f"\n[System] Model saved successfully to:\n{model_path}")
    
    return {'y': y_true_all, 'SPRA': spra_probs_all, 'NLI': nli_probs_all, 'Cosine': cos_probs_all, 'preds': spra_preds, 'drops': drops}, spra

def generate_visualizations(data, spra_instance, drops):
    y = data['y']
    def save_open(name):
        plt.savefig(name, dpi=300, bbox_inches='tight')
        print(f"Saved {name}")
        try: os.startfile(name)
        except: pass

    # FIG 1: Comparative ROC
    plt.figure(figsize=(8, 6))
    for name, color in [('SPRA', '#c44e52'), ('NLI', '#4c72b0'), ('Cosine', 'gray')]:
        fpr, tpr, _ = pd.Series(data[name]).pipe(lambda x: [0, 1, 1]) if len(np.unique(y))<2 else [0,0,0] # fallback
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, data[name])
        auc_v = roc_auc_score(y, data[name])
        plt.plot(fpr, tpr, color=color, linewidth=3 if name=='SPRA' else 1.5, label=f"{name} (AUC={auc_v:.2f})")
    plt.plot([0,1],[0,1], 'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("Fig 1. Comparative ROC")
    save_open('Fig1_ROC.png')

    # FIG 2: Precision-Recall
    plt.figure(figsize=(8, 6))
    for name, color in [('SPRA', '#c44e52'), ('NLI', '#4c72b0'), ('Cosine', 'gray')]:
        p, r, _ = precision_recall_curve(y, data[name])
        ap = average_precision_score(y, data[name])
        plt.plot(r, p, color=color, linewidth=3 if name=='SPRA' else 1.5, label=f"{name} (AP={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Fig 2. Precision-Recall Curve")
    save_open('Fig2_PR_Curve.png')

    # FIG 3: Calibration
    plt.figure(figsize=(6, 6))
    prob_true, prob_pred = calibration_curve(y, data['SPRA'], n_bins=5)
    plt.plot(prob_pred, prob_true, "s-", color='#c44e52', label='SPRA')
    plt.plot([0,1],[0,1], 'k--'); plt.legend(); plt.title("Fig 3. Calibration")
    save_open('Fig3_Calibration.png')

    # FIG 4: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y, data['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Hallucination'], yticklabels=['Safe', 'Hallucination'])
    plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.title("Fig 4. Confusion Matrix (SPRA)")
    save_open('Fig4_Confusion_Matrix.png')

    # FIG 5: Risk Separability
    plt.figure(figsize=(8, 5))
    sns.histplot(data['SPRA'][y==0], color='green', label='Safe', kde=True, stat="density", alpha=0.4)
    sns.histplot(data['SPRA'][y==1], color='red', label='Hallucination', kde=True, stat="density", alpha=0.4)
    plt.xlabel("Risk Score"); plt.title("Fig 5. Risk Separability"); plt.legend()
    save_open('Fig5_Risk_Dist.png')
    
    # FIG 6: Ablation Plot (from drops)
    plt.figure(figsize=(8, 5))
    plt.bar(drops.keys(), drops.values(), color='#8172b3', edgecolor='black')
    plt.ylabel("AUC Drop")
    plt.title("Fig 6. Feature Ablation Study")
    save_open('Fig6_Ablation.png')

# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    print("[1] Loading Data...")
    try:
        ds = load_dataset("aporia-ai/rag_hallucinations")
        df = pd.DataFrame(ds['train'])
        if len(df) > 2000: df = df.sample(2000, random_state=42)
        
        # Balance
        bad = df[df['is_hallucination']==True]
        good = df[df['is_hallucination']==False].sample(len(bad), random_state=42)
        df = pd.concat([bad, good]).sample(frac=1, random_state=42)
        
        # Run
        plot_data, spra_instance = run_benchmark(df)
        generate_visualizations(plot_data, spra_instance, plot_data['drops'])
        
        # Interactive
        print("\n" + "="*60)
        print(" INTERACTIVE HAZARD CHECK")
        print("="*60)
        while True:
            u = input("\nQuery: ")
            if u=='exit': break
            c = input("Context: ")
            risk, feats, reason = spra_instance.predict_hybrid(u, c)
            status = "✅ SAFE" if risk < 0.5 else "❌ HAZARD"
            print(f"Verdict: {status} (Risk: {risk:.4f})")
            print(f"Driver:  {reason}")

    except Exception as e: print(e)

if __name__ == "__main__":
    main()