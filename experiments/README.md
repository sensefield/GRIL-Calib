# 検証環境のセットアップと実行手順

PANDAR XT-32（spinning LiDAR, 32 line）+ IMU（200 Hz）を載せた地上車両の mcap bag に対して、
GRIL-Calib で LiDAR-IMU 外部キャリブレーションを実行し、結果を評価するための手順。

- LiDAR: `/sensing/lidar/top/pointcloud_raw_ex`（10 Hz）
- IMU: `/sensing/imu/imu_data`（200 Hz）

---

## 1. セットアップ

### 1.1 GRIL-Calib 本体

ビルドはリポジトリ直下の [README.md](../README.md) の Docker 手順をそのまま使う。要点:

1. `docker/` で `docker build -t gril-calib-ros2 .`
2. `./container_run.sh gril-calib-container gril-calib-ros2:latest`
3. コンテナ内で `livox_ros_driver2` をビルド → `colcon build --symlink-install` → `source install/setup.bash`

> 本検証は `feat/pandar` ブランチを使う（PANDAR XT-32 対応とクラッシュガードの修正が入っている）。
> ワークスペース側で `git checkout feat/pandar` してからビルドする。

### 1.2 評価スクリプトの依存

[tools/](tools/) の Python スクリプトは ROS2 (humble) 環境を source した状態で実行する。

- `rclpy`, `rosbag2_py`, `sensor_msgs`, `tf2_msgs`（ROS2 humble 同梱）
- mcap 読み込み用 `rosbag2_storage_mcap`（humble 同梱）
- `numpy`, `matplotlib`

```bash
pip install numpy matplotlib
```

---

## 2. 実行手順

### 2.1 設定ファイル

設定は [config/velodyne32.yaml](../config/velodyne32.yaml)。主なキー:

| キー | 値 | 意味 |
|---|---|---|
| `lid_topic` | `/sensing/lidar/top/pointcloud_raw_ex` | LiDAR 点群トピック |
| `imu_topic` | `/sensing/imu/imu_data` | IMU トピック |
| `lidar_type` | `5` | **5 = PANDAR**（`common_lib.h` の `LID_TYPE`。YAML のコメント "Velodyne" は古い表記） |
| `scan_line` | `32` | スキャンライン数 |
| `mean_acc_norm` | `9.843` | IMU の実測重力ノルム |
| `imu_sensor_height` | `1.2` | 地面〜IMU 高さ (m) |
| `data_accum_length` | `300.0` | データ蓄積しきい値 |
| `z_accumulate` | `0.5` | Z 軸回転の蓄積基準 |
| `bound_th` | `10.0` | translation 最適化の境界 (m) |
| `set_boundary` | `true` | 上記境界を有効化 |
| `gyro_factor` / `acc_factor` / `ground_factor` | `10 / 1 / 5` | 残差の重み |

### 2.2 GRIL-Calib の実行

コンテナ内（`ros2_ws` を source 済み）で:

```bash
# ターミナル A: GRIL-Calib ノード起動
ros2 launch gril_calib mapping_velodyne.launch.py
```

```bash
# ターミナル B: bag を t=0 から再生
ros2 bag play /path/to/your_bag.mcap
```

`data_sufficient` 条件を満たすと calibration が自動終了し、結果が標準出力に表示される。
`--start-offset` / `--duration` は付けず、bag 全体を t=0 から流す。

### 2.3 結果の確認

- キャリブ結果（回転行列・並進・time lag・bias）はノードの標準出力に出る。
- LiDAR オドメトリ軌跡は `trajectory_save.traj_file_path`（既定 `…/result/traj.txt`）に保存される。

### 2.4 評価スクリプト

すべて [tools/](tools/) 配下。`<BAG>` は対象 mcap の絶対パス、`<RESULT>` は 2.3 の標準出力を保存したファイル。

**(a) ground truth (`/tf_static`) との比較**

```bash
python3 tools/compare_with_tf_static.py <BAG> \
  --our-result <RESULT> \
  --out ground_truth_compare.md
```

**(b) キャリブに適した時間窓の解析（旋回活性度）**

```bash
python3 tools/analyze_bag_calibration_window.py <BAG> \
  --topic /sensing/imu/imu_data \
  --out window_analysis.md \
  --plot window_analysis.png
```

**(c) IMU 軸向きの判定**

```bash
python3 tools/step_a2_dynamic_segment_check.py <BAG> \
  --odom-topic /localization/pose_twist_fusion_filter/kinematic_state \
  --max-duration 650 \
  --dyn-yaw-th 0.10 --dyn-acc-th 0.40 --dyn-pad 0.4 \
  --out step_a2_kinematic_seg_650s.md
```

---

## 3. 実行上の注意

| ルール | 補足 |
|---|---|
| bag は t=0 から再生する | 途中スタートは AHRS のジャイロバイアス推定が温まらず、Rotation がずれる |
| `bound_th >= 4.0` にする | 小さいと translation X が最適化境界に張り付く |
| `data_accum_length` は `300` 程度に保つ | 大きくしすぎると時刻同期の相互相関が副ピークを誤検出する |
