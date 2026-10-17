"""
pocket_radius.py - measure a binding-pocket radius from a PDB entry (Colab-ready; needs internet and numpy only).

Usage (Google Colab):   !python pocket_radius.py 6UT0 6NJS --json pocket_radius_results.json
Reported in the manuscript (chain A, 4.5 A cutoff): 6UT0 (ligand M1X) R_median = 0.72 nm; 6NJS (ligand KQV) R_median = 0.83 nm.
For each entry the script downloads the coordinate file from RCSB, takes the largest non-solvent, non-ion ligand
(or the ligand named with --lig), finds protein heavy atoms within CUTOFF angstrom of any ligand atom (the pocket-lining atoms),
and reports, in nm:
   R_median = median distance from the ligand centroid to the pocket-lining atoms   ("mid radius")
   R_max    = largest such distance
   R_gyr    = radius of gyration of the ligand
State in the manuscript which measure was used, the cutoff, the chain and the ligand code.
"""
import sys, argparse, urllib.request
import numpy as np

CUTOFF = 4.5                                   # angstrom
SKIP = {"HOH", "WAT", "NA", "MG", "ZN", "CL", "K", "CA", "FMT", "DMS", "GOL", "EDO", "SO4", "PO4", "ACT", "PEG",
        "GDP", "GTP", "GNP", "GCP", "ACY", "TRS", "MPD", "PGE", "NO3", "IOD", "BR", "MN", "CD", "NI", "CO", "FE", "CU"}

def read_pdb(text):
    prot, het = [], []
    for ln in text.splitlines():
        if ln.startswith(("ATOM  ", "HETATM")):
            el = ln[76:78].strip() or ln[12:14].strip()
            if el.upper() == "H":
                continue
            rec = dict(name=ln[12:16].strip(), res=ln[17:20].strip(), chain=ln[21], resi=ln[22:27].strip(),
                       xyz=np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]), alt=ln[16])
            if rec["alt"] not in (" ", "A"):
                continue
            (prot if ln.startswith("ATOM  ") else het).append(rec)
    return prot, het

def analyse(pdb_id, lig=None, chain=None, cutoff=CUTOFF):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    text = urllib.request.urlopen(url, timeout=60).read().decode()
    prot, het = read_pdb(text)
    groups = {}
    for a in het:
        if a["res"] in SKIP and a["res"] != lig:
            continue
        groups.setdefault((a["res"], a["chain"], a["resi"]), []).append(a)
    if not groups:
        raise SystemExit(f"{pdb_id}: no ligand found")
    cand = [k for k in groups if (lig is None or k[0] == lig) and (chain is None or k[1] == chain)]
    if not cand:
        raise SystemExit(f"{pdb_id}: no ligand matches --lig/--chain")
    key = max(cand, key=lambda k: len(groups[k]))
    L = np.array([a["xyz"] for a in groups[key]])
    P = [a for a in prot if a["chain"] == key[1]] or prot
    PX = np.array([a["xyz"] for a in P])
    d = np.sqrt(((PX[:, None, :] - L[None, :, :]) ** 2).sum(-1)).min(axis=1)
    lining = PX[d <= cutoff]
    c = L.mean(axis=0)
    dist = np.sqrt(((lining - c) ** 2).sum(-1))
    out = dict(entry=pdb_id.upper(), ligand=key[0], chain=key[1], n_ligand_atoms=len(L), n_lining_atoms=len(lining),
               R_median_nm=float(np.median(dist) / 10), R_max_nm=float(dist.max() / 10),
               R_gyr_nm=float(np.sqrt(((L - c) ** 2).sum(-1).mean()) / 10))
    out["cutoff_A"] = cutoff
    return out

if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser(description="Measure a binding-pocket radius from PDB entries.")
    ap.add_argument("ids", nargs="+", help="PDB IDs, e.g. 6UT0 6NJS")
    ap.add_argument("--lig", default=None, help="three-letter ligand code (default: largest ligand)")
    ap.add_argument("--chain", default=None, help="chain ID (default: chain of the largest ligand)")
    ap.add_argument("--cutoff", type=float, default=CUTOFF, help="pocket-lining cutoff in angstrom (default 4.5)")
    ap.add_argument("--json", default=None, help="write all results to this JSON file")
    a = ap.parse_args()
    allres = []
    for i in a.ids:
        r = analyse(i, a.lig, a.chain, a.cutoff); allres.append(r)
        print(f"{r['entry']}: ligand {r['ligand']} (chain {r['chain']}, {r['n_ligand_atoms']} atoms), {r['n_lining_atoms']} lining atoms within {r['cutoff_A']} A;"
              f" R_median = {r['R_median_nm']:.2f} nm, R_max = {r['R_max_nm']:.2f} nm, ligand R_gyr = {r['R_gyr_nm']:.2f} nm")
    if a.json:
        json.dump(allres, open(a.json, "w"), indent=1)
        print("written:", a.json)
