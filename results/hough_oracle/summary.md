# Hough Oracle 摘要

可读帧：82；组合：108；耗时 2775.1s

论文固定自定义 Hough（Canny 35/110）：0.5s **0/82**

## 测试集 oracle 最优

- method: HOUGH_GRADIENT
- dp=1.2, minDist=15.0, param1=40.0, param2=20.0, minRadius=30, maxRadius=700
- 0.5 s: **0/82**
- 1.0 s: 0/82
- 2.0 s: 0/82
- 不测圆心: 0/82

说明：参数在同一测试集上挑最优，是乐观上界（oracle），不是独立验证集调参。
