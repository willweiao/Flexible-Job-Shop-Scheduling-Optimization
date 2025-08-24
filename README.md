# Flexible-Job-Shop-Scheduling-Optimization

使用数据集：Brandimarte FJSP Dataset (Credit to the repository: [SchedulingLab/fjsp-instances](https://github.com/SchedulingLab/fjsp-instances))

模型参考：[柔性车间作业调度（Flexible Job-shop Scheduling Problem, FJSP）](https://blog.csdn.net/weixin_46225503/article/details/132246053)

完成进度：

08/24 完成简化模型的建模以及对case mk01的求解、可视化

08/25 修改了模型存在的问题，主要是原模型缺少约束使每台机器上同一时间最多只有一个工序在加工，导致模型直接得到了LP松弛的最佳下界。修改后的模型可以计算出一个近优的可行解来。但是现在问题是无法有效地推进下界使得Gurobi的求解过程收敛，虽然在time limit内可以对mk01和mk02找到一个几乎最优解来，但是无法证明其最优性。同时，在解问题mk03和mk04时，因为规模的加大求解时间大幅增加，从而导致在time limit中，模型已无法找到足够多的可行解了，因而无法再有效了。需要对模型进行进一步的改良。

未来计划：

- 完成对全部15个case的求解与可视化
- 利用元启发式算法求解并与精确算法进行比较
- 增加其他可能的实际约束（due date, weight, 序列换型相关，物料到达窗口等）
- 其他可能的改善
