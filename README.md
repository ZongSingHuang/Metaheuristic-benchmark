# Metaheuristic-benchmark

# 1. 整理35篇期刊所使用的benchmark function(2007-2021)，並且統計各benchmark function的出現次數，詳見「Statistics table.xlsx」、「Figure.pdf」

# 2. 總計有69個benchmark function，函數名稱(function name)、邊界(bound)、理論最佳解(ideal solution, F*)、理論最佳解位置(ideal position, X*)、維度(Dimension, D)之參考依據請見「Fourmula.ipynb」

# 3. 除了Fletcher function因找不到公式完整定義，故無實現外，其餘68個benchmark function均已實現，詳見「benchmark.py」

# 4. 整理過程中有發現下列問題，為了統一，我把參照來源放在「Fourmula.ipynb」

4-1. 同一個benchmark function，每個作者所定義的邊界不相同

4-2. 同一個benchmark function，有多個名稱

4-3. benchmark function的公式，在細節上略有不同
