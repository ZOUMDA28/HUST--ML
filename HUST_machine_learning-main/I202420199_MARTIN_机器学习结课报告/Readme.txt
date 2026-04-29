机器学习大作业压缩包内容结构如下：
adaboost/
├── data/                                		 #存放测试数据data和targets
│   ├── data.csv
│   └── targets.csv
│
├── experiments/
│   └── base#_fold#                       #存放了四种数目基分类器的十折数据预测结果
│
├
├── adaboost.py             	 	# AdaBoost 算法实现
├── utils.py  				# 加载数据，k折算法
├── base_classifiers.py        	# 决策树桩基分类器算法实现与逻辑回归基分类器算法实现
├── main.py                       	 	# 运行入口
├── evaluate.py             		# 评测程序
├── __pycache__/            		# Python 缓存
│
├── Readme.txt                  		# 项目结构说明文档
├── 预测结果.xlsx                		# 不同的基函数预测结果
├── Adaboost_U202315600_田知恒_机器学习结课报告.docx             			# 结课报告



main.py的使用需要输入以下命令：
python main.py ./data/data.csv ./data/targets.csv [0/1]   #此处输入0代表使用对数逻辑回归作为基分类器，输入1代表使用决策树桩作为基分类器

至于环境，因为不需要用到独显，所以似乎cuda与conda等的冲突无关紧要，应该可以直接用集显的虚拟环境有sklearn和pandas等就行
