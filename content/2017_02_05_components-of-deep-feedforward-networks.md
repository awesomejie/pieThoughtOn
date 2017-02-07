Title: Components of Deep Feedforward Networks
Date: 2017-2-5 13:52
Category:
Tags: deep learning
Slug: components-of-deep-feedforward-networks
Authors:

构造一个人工智能系统面临的很多问题，都可以归纳成将一个矢量映射到另一个矢量。一般来说，人类能轻松完成的识别和分类任务，计算机可以通过训练深度神经网络与人类匹敌。神经网络构建的非线性模型适合描述这样的映射，而且研究发现深度神经网络可以表示任何函数（严格来说是Borel measurable function)。今天我们来介绍深度前向网络中的组成模块。

<!-- PELICAN_END_SUMMARY -->

####激活函数（activation function）
大多前向神经网络推荐使用ReLU(rectified linear unit)为激活函数。ReLu是非线性函数，但变换效果近似线性，其保留了线性模型generalization较好的特性，也适合用梯度下降法进行优化。计算机科学中人们的一个共识是用最简的单元构造复杂系统的能力。神经网络的研究证明了基于rectified linear函数，我们可以近似任意复杂的universal 函数。

####代价函数（cost function）
大部分情况下，神经网络的训练基于最大似然，因而代价函数即可表示为negative log-likelihood，等同于最小化训练数据和模型预测分布的交叉熵
$$
J(\boldsymbol\theta) = -\mathbb{E}_{\mathbf x,\mathbf y\sim\hat p_\text{data}} \log p_\text{model}(\mathbf{y}|\mathbf{x})
$$

假设模型输出服从高斯分布，代价函数就简化成了均方误差（mean squared error）。

如果有无穷的数据训练，最小化均方误差计算出来的模型是对每一个输入$\mathbf x$预测输出$\mathbf y$的均值。不同的代价函数对应不同的统计预测，如果是最小化mean absolute error，生成的模型对每一次的输入$\mathbf x$预测$\mathbf y$的median value。

####输出单元（output units）
我们一般用交叉熵（cross-entropy）作为代价函数，随后输出单元的选择则定义了cross-entropy的具体形式。假设前向网络的隐藏特征（hidden features）为$\mathbf h=f(\mathbf x;\boldsymbol \theta)$，输出层的作用是对$\mathbf h$进行最后一次转换以完成预测。

#####*线性单元*
线性输出层计算$\hat{\mathbf y} = \mathbf W^T \mathbf h + \mathbf b$。对于高斯分布$p(\mathbf y|\mathbf x) = \mathcal N(\mathbf y; \hat{\mathbf y},\mathbf I)$，最大似然等价于最小化交叉熵，最后简化为最小化均方误差。

#####*Sigmoid单元*
对于二元输出，最大似然定义输出服从Bernoulli分布。使用sigmoid作为输出函数，在最大似然的准则下我们得到的代价函数非常适合用梯度下降法优化，因为只有在模型得到最优解时，代价函数才饱和(saturate)。

#####*Softmax单元*
假如预测输出有$n$种取值，我们可用softmax函数表示$n$个取值每一种的概率。结合多元输出服从multinoulli分布，我们一样得到适合梯度下降优化的代价函数。二元条件下，等同于如上描述的sigmoid函数结合Bernoulli分布。

####隐藏单元（hidden units）
隐藏单元一般推荐ReLU。在人们发现ReLU之前，主要使用的是logistic sigmoid激活函数，或hyperbolic tangent激活函数。隐藏单元的设计是研究的热点，还有更多有效的函数尚待发掘。

####架构设计（architecture design）
前向网络的结构一般是分层链接的，设计一个网络主要需要选择网络的深度（层数）和每一层的宽度（结点数）。实验和研究验证，相对于浅层网络，层数多的网络的学习能力更强。

####梯度求解算法
优化网络的代价函数，能够有效计算梯度是优化的核心，因此人们发明了back-propagation算法，often simply called backprop。Back-propagation并不是唯一计算梯度的方法，但在神经网络的优化中被发现非常实用。一旦将来人们在自动差分领域有了进一步突破，神经网络梯度求解的性能还能提升。

####结语
利用梯度下降法最小化代价函数，前向神经网络可以有效的近似非线性函数。从这点来看，现代前向神经网络的发明和再发现是人类几个世纪求解general function近似问题研究和探索的顶点。
