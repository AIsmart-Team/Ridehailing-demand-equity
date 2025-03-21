# 考虑社会公平性的网约车需求预测
本文章提出了一个考虑社会公平性的GCN-LSTM网络对成都市市区范围内网约车的需求进行预测，其中“公平性”主要体现在在原本考虑预测准确性的损失函数（MAE）中加入衡量公平性的指标（MPE）
本次论文也是作者在深度学习方面的初次尝试，如果有相关的疑问，欢迎提出！

# Predicting Ride-Hailing Demand with Consideration of Social Equity

## Overview

本仓库包含了一种新颖的网约车需求预测方法的实现及研究成果，该方法特别考虑了社会公平因素。通过中国成都的案例研究，我们展示了如何通过结合人口统计、空间和交通可达性信息，在不同城市区域中实现更公平和更准确的需求预测。

## Motivation

共享出行服务已经改变了城市交通，但其带来的好处并不总是公平分配的。传统的需求预测模型通常在交通资源丰富的优势区域表现较好，而在服务不足的地区表现较差。这种差异可能会延续甚至加剧交通资源获取方面的现有社会不平等问题。
我们的研究通过将社会属性和人口特征明确纳入预测框架，解决了这一差距，从而提高了社会经济多样化城市区域的预测准确性和公平性。

## Key Contributions

开发了一种社会图卷积长短期记忆网络（SGC-LSTM）框架，将社会公平因素纳入网约车需求预测中，整合了多个功能图，包括：区域间的功能相似性；人口结构和人口统计信息；历史需求模式，在损失函数中引入平均百分比误差指标，以平衡预测准确性和公平性，对不同城市区域的预测准确性和公平性进行了全面评估，使用中国成都的真实数据进行了案例研究应用

## Methodology
我们的方法结合了图卷积网络和LSTM，以捕捉网约车需求的空间依赖性和时间模式。创新之处在于我们构建并整合了多个功能图：
功能相似性图：基于土地利用模式和兴趣点捕捉区域间的相似性
人口结构图：基于人口特征表示区域间的人口统计联系
历史需求图：编码区域间历史网约车需求流动的模式
SGC-LSTM框架并行处理这些图，并整合其特征，以生成在不同城市区域中既准确又公平的预测结果。

## Citation

If you use this work in your research, please cite:

@article{chen2024predicting,
  title={Predicting Ride-Hailing Demand with Consideration of Social Equity: A Case Study of Chengdu.},
  author={Chen, Xinran and Tu, Meiting and Gruyer, Dominique and Shi, Tongtong},
  journal={Sustainability (2071-1050)},
  volume={16},
  number={22},
  year={2024}
}

## Keywords

Ride-hailing, Demand prediction, Social equity, Graph convolution networks, LSTM, Transportation fairness, Urban mobility, Sustainable transportation
