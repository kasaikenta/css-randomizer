# qCSS Randomizer

**Orthogonality- and weight-preserving random modification of HX/HZ for CSS-LDPC codes**

This Python tool generates random orthogonal sparse matrix pairs `(HX, HZ)` for Calderbank–Shor–Steane (CSS) quantum LDPC codes.  
It repeatedly applies **2×2 cross-switch operations** to `HX` and repairs `HZ` by solving a local **integer linear problem (ILP)** via Google's **OR-Tools**.  
The result is a new random pair that maintains both orthogonality and the original row/column weight distribution.

---

## ✨ Features

- Orthogonality-preserving random modification of CSS matrices  
- Row and column weight conservation  
- ILP-based exact repair using CP-SAT (Google OR-Tools)  
- Sparse implementation (SciPy CSR)  
- PNG and GIF visualization of the matrix evolution  

---

## 🧩 Example Usage

```bash
python3 CSS_randomizer.py
```

The script produces:
- `HX_final.png`, `HZ_final.png` — final sparse structure snapshots  
- `evolution.gif` — optional animation of HX/HZ evolution

---

## 📦 Requirements

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.26
scipy>=1.11
ortools>=9.8
matplotlib>=3.8
pillow>=10.0
```

---

## 🧠 Citation

If you use this tool in academic work, please cite:

> K. Kasai, *"Random Orthogonality-Preserving Construction for CSS-LDPC Codes"*, 2025.

---

## ⚖️ License

This project is distributed under the MIT License.  
See [LICENSE](LICENSE) for details.

---

## 🧪 Repository structure

```
qcss-randomizer/
├── CSS_randomizer.py     # main script (orthogonality-preserving randomizer)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧬 Contact

**Kenta Kasai**  
Institute of Science Tokyo (Tokyo Tech)  
GitHub: [@kasaikenta](https://github.com/kasaikenta)
