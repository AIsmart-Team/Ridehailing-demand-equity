# Ridehailing-demand-equity
A novel Social Graph Convolution LSTM framework for predicting ride-hailing demand with enhanced social equity considerations, demonstrated through a case study of Chengdu, China.

# Predicting Ride-Hailing Demand with Consideration of Social Equity

## Overview

This repository contains the implementation and research findings for a novel approach to ride-hailing demand prediction that specifically considers social equity factors. Using a case study from Chengdu, China, we demonstrate how incorporating demographic, spatial, and transportation accessibility information can lead to more equitable and accurate predictions across diverse urban regions.

## Motivation

Shared mobility services have transformed urban transportation, but their benefits are not always distributed equitably. Conventional demand prediction models often achieve higher accuracy in privileged areas with abundant transportation resources while performing poorly in underserved regions. This disparity can perpetuate and amplify existing social inequities in transportation access.

Our research addresses this gap by explicitly incorporating social attributes and demographic characteristics into the prediction framework, thereby improving both prediction accuracy and fairness across socioeconomically diverse urban areas.

## Key Contributions

- Development of a **Social Graph Convolution Long Short-Term Memory (SGCLSTM)** framework that incorporates social equity considerations into ride-hailing demand prediction
- Integration of multiple functional graphs representing:
  - Functional similarity between regions
  - Population structure and demographic information
  - Historical demand patterns
- Implementation of Mean Percentage Error indicators in the loss function to balance prediction accuracy and fairness
- Comprehensive evaluation of both prediction accuracy and equity across diverse urban regions
- Case study application using real-world data from Chengdu, China

## Methodology

Our approach combines graph convolutional networks with LSTM to capture both spatial dependencies and temporal patterns in ride-hailing demand. The innovation lies in how we construct and integrate multiple functional graphs:

1. **Functional Similarity Graph**: Captures the similarity between regions based on land use patterns and points of interest
2. **Population Structure Graph**: Represents demographic connections between regions based on population characteristics
3. **Historical Demand Graph**: Encodes patterns of historical ride-hailing demand flows between regions

The SGCLSTM framework processes these graphs in parallel and integrates their features to generate predictions that are both accurate and fair across different urban regions.

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
