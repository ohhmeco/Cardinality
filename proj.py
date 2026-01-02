import math
import randomhash
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Genera file senza aprire finestre
import matplotlib.pyplot as plt

class HyperLogLog:
    def __init__(self, p):
        self.p = p
        self.m = 1 << p
        self.M = [0] * self.m
        self.rfh = randomhash.RandomHashFamily(count=1)
        if self.m == 16: self.alpha_m = 0.673
        elif self.m == 32: self.alpha_m = 0.697
        elif self.m == 64: self.alpha_m = 0.709
        else: self.alpha_m = 0.7213 / (1 + 1.079 / self.m)

    def _get_rho(self, w, max_bits):
        binary_w = bin(w)[2:].zfill(max_bits)
        idx = binary_w.find('1')
        return idx + 1 if idx != -1 else max_bits + 1

    def add(self, item):
        x = self.rfh.hash(str(item))
        j = x & (self.m - 1)
        w = x >> self.p
        self.M[j] = max(self.M[j], self._get_rho(w, 32 - self.p))

    def estimate(self):
        sum_registers = sum(2.0**-x for x in self.M)
        E = self.alpha_m * (self.m**2) * (1.0 / sum_registers)
        if E <= 2.5 * self.m:
            V = self.M.count(0)
            if V > 0: E = self.m * math.log(self.m / V)
        return int(E)

class Recordinality:
    def __init__(self, k):
        self.k = k
        self.S = set()
        self.R = 0
        self.rfh = randomhash.RandomHashFamily(count=1)
        self.max_val = None

    def add(self, item):
        h = self.rfh.hash(str(item))
        if len(self.S) < self.k:
            self.S.add(h)
            if self.max_val is None or h > self.max_val: self.max_val = h
        elif h not in self.S:
            if h < self.max_val:
                self.S.remove(self.max_val)
                self.S.add(h)
                self.R += 1
                self.max_val = max(self.S)

    def estimate(self):
        return int(self.k * math.pow(1 + 1/self.k, self.R))



def get_exact_count(dat_path):
    try:
        with open(dat_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except: return None

def generate_zipf_stream(n_distinct, N_total, alpha):
    ranks = np.arange(1, n_distinct + 1)
    weights = 1 / (ranks ** alpha)
    weights /= weights.sum()
    return np.random.choice(ranks, size=N_total, p=weights)




def main():
    if not os.path.exists("Plots"): os.makedirs("Plots")
    
    p_def, k_def, T = 10, 1024, 10
    dataset_dir = "datasets"
    
    dataset_files = []
    if os.path.exists(dataset_dir):
        for f in sorted(os.listdir(dataset_dir)):
            if f.endswith(".txt"):
                base = f.replace(".txt", "")
                dat_p = os.path.join(dataset_dir, f.replace(".txt", ".dat"))
                if os.path.exists(dat_p):
                    dataset_files.append((base, os.path.join(dataset_dir, f), dat_p))

    if not dataset_files:
        print("Errore: Nessun file .txt/.dat trovato in 'datasets/'")
        return

    global_stats = {b[0]: {'r': 0, 'h': 0, 'c': 0} for b in dataset_files}
    global_stats['Zipf_Alpha_1.0'] = {'r': 0, 'h': 0, 'c': 0}

    header = f"{'Dataset':<20} | {'Reale':>8} | {'HLL':>8} | {'ErrH%':>7} | {'REC':>8} | {'ErrR%':>7}"
    
    print("\n" + "="*90)
    print(f"{'FASE 1: ESECUZIONE 10 ESPERIMENTI COMPARATIVI':^90}")
    print("="*90)

    for exp_id in range(1, T + 1):
        print(f"\n[ESPERIMENTO {exp_id}/{T}]")
        print(header)
        print("-" * len(header))
        
        for name, txt, dat in dataset_files:
            real = get_exact_count(dat)
            h, r = HyperLogLog(p_def), Recordinality(k_def)
            with open(txt, 'r', encoding='utf-8') as f:
                for line in f:
                    for w in re.findall(r'\w+', line.lower()): h.add(w); r.add(w)
            
            eh, er = h.estimate(), r.estimate()
            ph, pr = (eh-real)/real*100, (er-real)/real*100
            print(f"{name:<20} | {real:>8} | {eh:>8} | {ph:>6.2f}% | {er:>8} | {pr:>6.2f}%")
            
            global_stats[name]['r'] += real; global_stats[name]['h'] += eh; global_stats[name]['c'] += er

        # Zipf reference per ogni esperimento
        s = generate_zipf_stream(10000, 100000, 1.0)
        rz = len(set(s))
        hz, rz_est = HyperLogLog(p_def), Recordinality(k_def)
        for x in s: hz.add(x); rz_est.add(x)
        ezh, ezr = hz.estimate(), rz_est.estimate()
        print(f"{'Zipf_Alpha_1.0':<20} | {rz:>8} | {ezh:>8} | {(ezh-rz)/rz*100:>6.2f}% | {ezr:>8} | {(ezr-rz)/rz*100:>6.2f}%")
        global_stats['Zipf_Alpha_1.0']['r'] += rz; global_stats['Zipf_Alpha_1.0']['h'] += ezh; global_stats['Zipf_Alpha_1.0']['c'] += ezr

    # final average
    print("\n" + "#"*90)
    print(f"{'RESOCONTO FINALE MEDIATO (T=10)':^90}")
    print("#"*90)
    print(header)
    for name, d in global_stats.items():
        ar, ah, ac = d['r']/T, d['h']/T, d['c']/T
        print(f"{name:<20} | {ar:>8.1f} | {ah:>8.1f} | {(ah-ar)/ar*100:>6.2f}% | {ac:>8.1f} | {(ac-ar)/ar*100:>6.2f}%")

# Memory, added Rec Theory Error 
    print("\n" + "="*90)
    print(f"{'FASE 2: ANALISI MEMORIA (DRACULA) - CONFRONTO TEORICO HLL vs REC':^90}")
    print("="*90)
    mem_header = f"{'p':<4} | {'m/k':>6} | {'Theo HLL%':>10} | {'Theo REC%':>10} | {'Emp HLL%':>10} | {'Emp REC%':>10}"
    print(mem_header)
    print("-" * len(mem_header))
    
    p_vals = [6, 8, 10, 12]
    hll_m_errs, rec_m_errs = [], []
    theo_hll_vals, theo_rec_vals = [], []
    _, drac_txt, drac_dat = dataset_files[0]
    real_drac = get_exact_count(drac_dat)

    for p in p_vals:
        m = 2**p
        h_tmp, r_tmp = [], []
        for _ in range(5):
            h, r = HyperLogLog(p), Recordinality(m)
            with open(drac_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    for w in re.findall(r'\w+', line.lower()): h.add(w); r.add(w)
            h_tmp.append(abs(h.estimate()-real_drac)/real_drac*100)
            r_tmp.append(abs(r.estimate()-real_drac)/real_drac*100)
        
        avg_h, avg_r = np.mean(h_tmp), np.mean(r_tmp)
        t_hll = (1.04 / math.sqrt(m)) * 100
        t_rec = (1.0 / math.sqrt(m)) * 100 # Errore teorico REC: 1/sqrt(k)
        
        print(f"{p:<4} | {m:>6} | {t_hll:>9.2f}% | {t_rec:>9.2f}% | {avg_h:>9.2f}% | {avg_r:>9.2f}%")
        
        hll_m_errs.append(avg_h); rec_m_errs.append(avg_r)
        theo_hll_vals.append(t_hll); theo_rec_vals.append(t_rec)

    # Plot Memory
    plt.figure(figsize=(10, 6))
    plt.plot([2**p for p in p_vals], theo_hll_vals, 'b--', label='Theory HLL ($1.04/\\sqrt{m}$)', alpha=0.6)
    plt.plot([2**p for p in p_vals], theo_rec_vals, 'r--', label='Theory REC ($1/\\sqrt{k}$)', alpha=0.6)
    plt.plot([2**p for p in p_vals], hll_m_errs, 'b-o', label='Empirical HLL', linewidth=2)
    plt.plot([2**p for p in p_vals], rec_m_errs, 'r-s', label='Empirical REC', linewidth=2)
    
    plt.xscale('log', base=2); plt.grid(True, which="both", ls="--"); plt.legend()
    plt.xlabel('Memory (m/k)'); plt.ylabel('Relative Error (%)'); plt.title('Memory Study: Theory vs Empirical Convergence')
    plt.savefig('Plots/grafico_memoria.png'); plt.close()

    # ANALYSIS ALPHA (ZIPF)
    print("\n" + "="*90)
    print(f"{'FASE 3: ANALISI IMPATTO SKEWNESS (Sintetico)':^90}")
    print("="*90)
    alpha_header = f"{'Alpha':<10} | {'Real Card':>10} | {'Err HLL%':>10} | {'Err REC%':>10}"
    print(alpha_header)
    print("-" * len(alpha_header))
    
    alphas = [0.5, 1.0, 1.5, 2.0]
    h_a_errs, r_a_errs = [], []
    for a in alphas:
        stream = generate_zipf_stream(5000, 50000, a)
        rz = len(set(stream))
        h, r = HyperLogLog(10), Recordinality(1024)
        for x in stream: h.add(x); r.add(x)
        eh_a = abs(h.estimate()-rz)/rz*100
        er_a = abs(r.estimate()-rz)/rz*100
        print(f"{a:<10} | {rz:>10} | {eh_a:>9.2f}% | {er_a:>9.2f}%")
        h_a_errs.append(eh_a); r_a_errs.append(er_a)

    # Plot alpha
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(alphas))
    plt.bar(x_pos - 0.2, h_a_errs, 0.4, label='HLL', color='skyblue', edgecolor='black')
    plt.bar(x_pos + 0.2, r_a_errs, 0.4, label='REC', color='salmon', edgecolor='black')
    plt.xticks(x_pos, alphas); plt.xlabel('Zipf Alpha (Skewness)'); plt.ylabel('Relative Error (%)'); plt.legend()
    plt.title('Alpha Impact Analysis')
    plt.savefig('Plots/grafico_alpha.png'); plt.close()

    print("\n" + "="*90)
    print("SIMULAZIONE COMPLETATA. Grafici salvati nella cartella 'Plots/'.")
    print("="*90)

if __name__ == "__main__":
    main()
