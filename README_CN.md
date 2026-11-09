# OrganPFN-M 代码公开包

本仓库整理了本研究的数据预处理、五个缺失阈值的自监督训练、五个阈值的
健康样本五折微调、PIOAA 器官衰老不同步性，以及器官衰老至疾病联级传播分析。
仓库不包含 UK Biobank 个体级数据、个体预测结果、训练权重或第三方 TabPFN 权重。

数据口径与论文一致：T0 生理检查和 T2 影像随访按独立样本行表示，同一参与者
可出现两行，而不是把两次访视横向拼接成一个样本。每行包含十个器官块、776 个
测量特征和唯一的样本级 Sex 字段。缺失器官及器官内缺失特征保留为空。分别按
器官内最大缺失比例 0%、30%、50%、70% 和 90% 构建数据，每个样本至少保留
两个器官。

论文后续分析采用阈值 90 的自监督检查点，使用阈值 30 的健康样本进行五折微调，
并采用非对称损失微调结果中的 fold 4 权重。公开脚本同时保留全部五个阈值和五个
折次，便于复现敏感性分析。

快速检查：

```bash
conda env create -f environment.yml
conda activate organpfnm
python examples/make_synthetic_data.py --output-dir examples/generated
python scripts/prepare_data.py --input examples/generated/harmonized_visits.csv \
  --feature-map examples/generated/feature_map.json \
  --output-root examples/generated/processed
python -m pytest
```

正式公开前请完成 `LICENSE_TO_SELECT.md` 和 `CITATION.cff` 中的机构许可、作者、
仓库地址与 DOI 信息，并再次确认 Git 历史中没有受限数据。
