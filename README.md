# test20251120

data processing helper for detecting stable voltage plateaus in CSV waveforms.

## 使用方法
1. 安装依赖：`pip install pandas numpy scipy matplotlib`
2. 运行示例（会生成 demo_plots 下的图像）：
   ```bash
   python stable_segments.py
   ```
3. 在自己的数据上使用：
   ```python
   from pathlib import Path
   from stable_segments import process_file, StabilityConfig

   segments, series = process_file(
       Path("1.csv"),
       Path("plots"),
       StabilityConfig(window_size=50, std_threshold=0.5, min_length=50, voltage_column="voltage"),
   )
   ```
   其中 `voltage_column` 指向 CSV 中的电压列名。每个稳定区间的正态分布图会保存到 `plots` 目录。
