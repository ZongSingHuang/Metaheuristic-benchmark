# Metaheuristic-benchmark

# 1. 抱怨:

1-1. 我對軟性計算非常有興趣，這點從我的Github上面的作品就可以察覺

1-2. 但是我發現，很多期刊都沒有完整描述每一測試函數的名稱、維度、解空間邊界、理論最佳適應值、理論最佳解，導致我在測試自己開發的演算法上造成阻礙

1-3. 舉個例子，光是最簡單的Sphere我就看到三種解空間邊界，甚至還有兩種公式，整個莫名其妙

1-4. 第二個例子，有些期刊只寫上這個測試函數叫做Schwefel，但就我知道的，一樣叫作Schwefel起碼有5個版本(有各自專屬名稱)。這就像你在辦公室裏面喊李先先收包裹，阿到底是哪個李先生???

# 2. 說明:

2-1. 為了自己方，也不想讓其他人跟我有一樣的遭遇，所以我用閒暇時間整理自己手邊35篇期刊至「活頁簿1.xlsx」，共計598個測試函數(不重複75)

2-2. 每一個測試函數我都有註記出處、名稱、維度、解空間邊界，理論最佳適應值、理論最佳解我則是寫在「benchmark.py」

2-3. 其中有2個測試函數找不到名稱故不實現，有1個測試函數因為缺少關鍵參數故無法實現，其餘73個測試函數均有實現

# 3. 參考文獻:

https://arxiv.org/abs/1308.4008  -> Xin-She Yang函數，同時也是Bat Algorithm的作者整理

http://benchmarkfcns.xyz/fcns  -> Mazhar Ansari Ardeh

https://www.al-roomi.org/benchmarks  -> University of Bahrain 實驗室整理

https://www.sfu.ca/~ssurjano/optimization.html  -> Simon Fraser University 實驗室整理

http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm

https://www.indusmic.com/blog/categories/benchmark-function  -> Kyoto University 實驗室整理

https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda#ackley-function  -> tomitomi3

https://en.wikipedia.org/wiki/Test_functions_for_optimization

https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/index.html  -> University of New Mexico 實驗室整理

01_Adaptive Particle Swarm Optimization

02_A novel bee swarm optimization algorithm for numerical function optimization

03_Chaos-enhanced accelerated particle swarm optimization

04_Experimental Analysis of Bound Handling Techniques in Particle Swarm Optimization

05_Biogeography-based optimisation with chaos

06_Grey Wolf Optimizer

07_The Whale Optimization Algorithm

08_Hybrid Algorithm of Particle Swarm Optimization and Grey Wolf Optimizer for Improving Convergence Performance

09_Lévy Flight Trajectory-Based Whale Optimization Algorithm for Global Optimization

10_Chaotic whale optimization algorithm

11_一种改进的鲸鱼优化算法

12_A hyper-heuristic for improving the initial population of whale optimization algorithm

13_An efficient double adaptive random spare reinforced whale optimization algorithm

14_An enhanced associative learning-based exploratory whale optimizer for global optimization

15_An Enhanced Whale Optimization Algorithm with Simplex Method

16_Chaotic particle swarm optimization with sigmoid-based acceleration coefficients for numerical function optimization

17_New binary whale optimization algorithm for discrete optimization problems

18_改进的鲸鱼优化算法及其应用

19_基于自适应权重和模拟退火的鲸鱼优化算法

20_基于改进型鲸鱼优化算法和最小二乘支持向量机的炼钢终点预测模型研究

21_精英反向黄金正弦鲸鱼算法及其工程优化研究

22_Enhanced whale optimization algorithm for maximum power point tracking of variable-speed wind generators

23_Improved Whale Optimization Algorithm applied to design PID plus second-order derivative controller for automatic voltage regulator system

24_Improved Whale Optimization Algorithm Based on Nonlinear Adaptive Weight and Golden Sine Operator

25_Multi-Strategy Ensemble Whale Optimization Algorithm and Its Application to Analog Circuits Intelligent Fault Diagnosis

26_Opposition based competitive grey wolf optimizer for EMG feature

27_一种离散鲸鱼算法及其应用

28 I-GWO and Ex-GWO improved algorithms of the Grey Wolf Optimizer to solve global optimization problems

29_A novel enhanced whale optimization algorithm for global optimization

30_Improved Whale Optimization Algorithm for Solving Constrained Optimization Problems

31_Nature-inspired approach An enhanced whale optimization algorithm for global optimization

32_A new mutation operator for real coded genetic algorithms

33_A new crossover operator for real coded genetic algorithms

34_An Efficient Real-coded Genetic Algorithm for RealParameter Optimization

35_An new crossover operator for real-coded genetic algorithm with selective breeding based on difference between individuals
