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

http://infinity77.net/global_optimization/genindex.html

http://benchmarkfcns.xyz/fcns

https://www.al-roomi.org/benchmarks

https://www.sfu.ca/~ssurjano/optimization.html

http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm

https://www.indusmic.com/blog/categories/benchmark-function

https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda#ackley-function

https://en.wikipedia.org/wiki/Test_functions_for_optimization

https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/index.html
