import sys
import os

# 解决打包后路径问题
if getattr(sys, 'frozen', False):
    # 如果是打包后的exe文件
    application_path = os.path.dirname(sys.executable)
else:
    # 如果是python脚本
    application_path = os.path.dirname(os.path.abspath(__file__))

# 添加当前目录到系统路径
sys.path.append(application_path)
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QFrame, QGroupBox, QStatusBar,
    QSlider, QComboBox, QTextEdit, QCheckBox, QTabWidget, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# 专业技术分析器
# ----------------------------
class TechniqueAnalyzer:
    def __init__(self):
        pass
    
    def evaluate_smash_technique(self, angles, hits, fps_data):
        """综合评估扣杀技术"""
        if len(angles) < 10:
            return {"总评": "数据不足", "建议": "继续练习以获取更多分析数据"}
        
        # 1. 引拍充分性评估
        backswing_score = self._evaluate_backswing(angles)
        
        # 2. 击球力量评估
        power_score = self._evaluate_power(hits, len(angles))
        
        # 3. 动作稳定性评估
        stability_score = self._evaluate_stability(angles)
        
        # 4. 节奏感评估
        rhythm_score = self._evaluate_rhythm(angles)
        
        # 计算总分
        total_score = (backswing_score + power_score + stability_score + rhythm_score) / 4
        
        # 生成评级
        grade = self._get_grade(total_score)
        
        return {
            '总评分数': round(total_score, 1),
            '技术等级': grade,
            '引拍质量': round(backswing_score, 1),
            '击球力量': round(power_score, 1),
            '动作稳定性': round(stability_score, 1),
            '节奏感': round(rhythm_score, 1),
            '详细建议': self._generate_detailed_recommendations(
                backswing_score, power_score, stability_score, rhythm_score
            )
        }
    
    def _evaluate_backswing(self, angles):
        """评估引拍质量"""
        if len(angles) < 10:
            return 50
            
        # 查找引拍过程中的最大角度
        max_angle = max(angles) if angles else 90
        # 160-170度为理想引拍角度
        if 160 <= max_angle <= 175:
            return 95
        elif 150 <= max_angle < 160 or 175 < max_angle <= 180:
            return 80
        elif 140 <= max_angle < 150 or 180 < max_angle <= 190:
            return 65
        else:
            return 40
    
    def _evaluate_power(self, hits, total_frames):
        """评估击球力量（基于击球频率）"""
        if total_frames == 0:
            return 50
            
        # 计算击球频率（每100帧的击球数）
        hit_frequency = (hits / total_frames) * 100
        
        if hit_frequency > 8:
            return 90
        elif hit_frequency > 5:
            return 75
        elif hit_frequency > 3:
            return 60
        else:
            return 45
    
    def _evaluate_stability(self, angles):
        """评估动作稳定性"""
        if len(angles) < 10:
            return 50
            
        # 计算角度的标准差，越小越稳定
        std_dev = np.std(angles)
        
        if std_dev < 10:
            return 90
        elif std_dev < 20:
            return 75
        elif std_dev < 30:
            return 60
        else:
            return 40
    
    def _evaluate_rhythm(self, angles):
        """评估节奏感"""
        if len(angles) < 30:
            return 50
            
        # 简单节奏评估：检查角度变化的规律性
        # 计算相邻角度变化的方差
        diffs = [abs(angles[i+1] - angles[i]) for i in range(len(angles)-1)]
        rhythm_var = np.var(diffs)
        
        if rhythm_var < 25:
            return 85
        elif rhythm_var < 50:
            return 70
        elif rhythm_var < 100:
            return 55
        else:
            return 40
    
    def _get_grade(self, score):
        """根据总分获取技术等级"""
        if score >= 90:
            return "优秀 (A)"
        elif score >= 80:
            return "良好 (B)"
        elif score >= 70:
            return "中等 (C)"
        elif score >= 60:
            return "及格 (D)"
        else:
            return "待提高 (F)"
    
    def _generate_detailed_recommendations(self, backswing, power, stability, rhythm):
        """生成详细的改进建议"""
        recommendations = []
        
        if backswing < 70:
            recommendations.append("🔹 增加引拍幅度，确保充分蓄力")
        elif backswing < 85:
            recommendations.append("🔸 引拍幅度适中，可进一步优化")
        else:
            recommendations.append("✅ 引拍动作标准")
            
        if power < 70:
            recommendations.append("🔹 提高击球频率，增强爆发力")
        elif power < 85:
            recommendations.append("🔸 击球力量中等，有提升空间")
        else:
            recommendations.append("✅ 击球力量充足")
            
        if stability < 70:
            recommendations.append("🔹 加强动作稳定性练习")
        elif stability < 85:
            recommendations.append("🔸 动作稳定性较好，继续保持")
        else:
            recommendations.append("✅ 动作非常稳定")
            
        if rhythm < 70:
            recommendations.append("🔹 注意动作节奏，保持连贯性")
        elif rhythm < 85:
            recommendations.append("🔸 动作节奏感良好")
        else:
            recommendations.append("✅ 动作节奏流畅")
            
        # 通用建议
        recommendations.extend([
            "\n📋 专业训练建议:",
            "- 每天进行30分钟的基础挥拍练习",
            "- 注重动作的连贯性和稳定性",
            "- 录制动作视频进行自我分析",
            "- 多球练习提高反应速度和准确性"
        ])
        
        return "\n".join(recommendations)

# ----------------------------
# 数据处理线程
# ----------------------------
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_plot_signal = pyqtSignal(float)
    error_signal = pyqtSignal(str)
    update_stats_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)  # 默认摄像头
        self.angles = []
        self.prev_angle = None
        self.swing_start = False
        self.frame_count = 0
        self.fps = 0
        self.hits = 0
        
        # 可调节参数
        self.hit_threshold_high = 150
        self.hit_threshold_low = 90
        self.swing_start_threshold_high = 150
        self.swing_start_threshold_low = 140
        self.sensitivity = 20  # 新增灵敏度参数
        
        # 新增用于改进检测的属性
        self.angle_history = []  # 角度历史记录
        self.min_angle_since_start = 180  # 引拍开始后的最小角度
        self.use_advanced_detection = True  # 是否使用高级检测算法

    def set_video_source(self, path):
        self.cap.release()
        self.cap = cv2.VideoCapture(path)

    def set_hit_thresholds(self, high, low):
        self.hit_threshold_high = high
        self.hit_threshold_low = low

    def set_swing_start_thresholds(self, high, low):
        self.swing_start_threshold_high = high
        self.swing_start_threshold_low = low

    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity

    def set_detection_mode(self, use_advanced):
        self.use_advanced_detection = use_advanced

    def detect_smash_advanced(self, current_angle):
        """改进的击球检测算法"""
        if current_angle is None:
            return False
        
        # 记录角度历史（最多保存10帧）
        self.angle_history.append(current_angle)
        if len(self.angle_history) > 10:
            self.angle_history.pop(0)
        
        # 需要至少3帧数据才能判断
        if len(self.angle_history) < 3:
            return False
        
        # 检查最近几帧的角度变化趋势
        recent_angles = self.angle_history[-3:]
        
        # 判断是否为快速下压动作：角度快速减小且变化率大
        if (len(recent_angles) >= 3 and 
            recent_angles[0] > recent_angles[1] > recent_angles[2] and
            (recent_angles[0] - recent_angles[2]) > self.sensitivity and  # 总角度变化
            (recent_angles[0] - recent_angles[1]) > 5 and   # 第一阶段变化
            (recent_angles[1] - recent_angles[2]) > 5):     # 第二阶段变化
            return True
        
        return False

    def run(self):
        import time
        start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                
                # 计算FPS
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    self.fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = time.time()
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                current_angle = None
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = frame.shape

                    try:
                        # 获取右臂关键点
                        RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                        RIGHT_ELBOW = self.mp_pose.PoseLandmark.RIGHT_ELBOW
                        RIGHT_WRIST = self.mp_pose.PoseLandmark.RIGHT_WRIST
                        
                        if all(landmark in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST] 
                               for landmark in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
                            
                            shoulder = np.array([
                                landmarks[RIGHT_SHOULDER].x * w,
                                landmarks[RIGHT_SHOULDER].y * h
                            ], dtype=int)
                            elbow = np.array([
                                landmarks[RIGHT_ELBOW].x * w,
                                landmarks[RIGHT_ELBOW].y * h
                            ], dtype=int)
                            wrist = np.array([
                                landmarks[RIGHT_WRIST].x * w,
                                landmarks[RIGHT_WRIST].y * h
                            ], dtype=int)

                            # 绘制关键点
                            cv2.circle(frame, tuple(shoulder), 8, (0, 255, 0), -1)
                            cv2.circle(frame, tuple(elbow), 8, (0, 255, 0), -1)
                            cv2.circle(frame, tuple(wrist), 8, (0, 255, 0), -1)
                            cv2.line(frame, tuple(shoulder), tuple(elbow), (255, 0, 0), 3)
                            cv2.line(frame, tuple(elbow), tuple(wrist), (255, 0, 0), 3)

                            # 计算夹角
                            ba = shoulder - elbow
                            bc = wrist - elbow
                            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                            current_angle = np.degrees(angle)
                            current_angle = round(current_angle, 1)

                            # 显示角度
                            cv2.putText(frame, f'{current_angle} deg', tuple(elbow),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                            # 显示FPS
                            cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # 显示引拍状态
                            if self.swing_start:
                                cv2.putText(frame, "SWING READY", (10, 70),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                cv2.putText(frame, f"Min Angle: {self.min_angle_since_start:.1f}", (10, 100),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                            # 挥拍判断 - 改进的算法
                            if self.use_advanced_detection:
                                # 更新引拍状态
                                if not self.swing_start and current_angle > self.swing_start_threshold_high:
                                    self.swing_start = True
                                    self.min_angle_since_start = current_angle
                                    logger.info("🔄 引拍开始")
                                
                                # 更新引拍过程中的最小角度
                                if self.swing_start:
                                    self.min_angle_since_start = min(self.min_angle_since_start, current_angle)
                                
                                # 检测击球
                                if self.swing_start and self.detect_smash_advanced(current_angle):
                                    self.hits += 1
                                    logger.info(f"💥 击球！角度：{current_angle}°")
                                    cv2.putText(frame, "HIT!", (100, 100),
                                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                                    self.swing_start = False  # 重置状态
                                    self.min_angle_since_start = 180
                                    self.angle_history.clear()  # 清除历史记录
                            else:
                                # 原始算法
                                if self.prev_angle is not None:
                                    if not self.swing_start and current_angle > self.swing_start_threshold_high and self.prev_angle < self.swing_start_threshold_low:
                                        self.swing_start = True
                                        logger.info("🔄 引拍开始")
                                    elif self.swing_start and current_angle < self.hit_threshold_low and self.prev_angle > self.hit_threshold_high:
                                        self.hits += 1
                                        logger.info(f"💥 击球！角度：{current_angle}°")
                                        cv2.putText(frame, "HIT!", (100, 100),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                                        self.swing_start = False

                            self.prev_angle = current_angle
                            self.angles.append(current_angle)
                            self.update_plot_signal.emit(current_angle)
                            
                            # 发送统计数据更新信号
                            stats = {
                                'current_angle': current_angle,
                                'fps': self.fps,
                                'hits': self.hits,
                                'avg_angle': np.mean(self.angles) if self.angles else 0
                            }
                            self.update_stats_signal.emit(stats)

                    except Exception as e:
                        logger.error(f"处理姿势数据时出错: {str(e)}")
                        self.error_signal.emit(f"处理姿势数据时出错: {str(e)}")

                # 发送图像到 GUI
                self.change_pixmap_signal.emit(frame)
            
            # 控制帧率，减少CPU占用
            self.msleep(30)

        self.cap.release()
        # 清理mediapipe资源
        self.pose.close()

    def stop(self):
        self.running = False
        self.wait()
        logger.info("视频线程已停止")


# ----------------------------
# 主窗口
# ----------------------------
class BadmintonAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏸 羽毛球动作分析系统")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #f0f0f0; font-family: Arial;")

        # 初始化线程
        self.thread = VideoThread()
        self.technique_analyzer = TechniqueAnalyzer()

        self.init_ui()
        self.setup_connections()
        
        # 初始化状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 初始化专业建议定时器
        self.advice_timer = QTimer()
        self.advice_timer.timeout.connect(self.update_professional_advice)
        self.analysis_started = False

    def init_ui(self):
        # 主布局
        container = QWidget()
        layout = QHBoxLayout(container)
        self.setCentralWidget(container)

        # 左侧：视频显示
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #ccc;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # 右侧：控制面板 + 图表
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setFixedWidth(380)

        # 标题
        title = QLabel("🏸 动作分析面板")
        title.setFont(QFont("SimHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)

        # 标签页控件
        self.tab_widget = QTabWidget()
        
        # 实时数据展示 tab
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)
        
        self.current_angle_label = QLabel("当前角度: - °")
        self.fps_label = QLabel("FPS: -")
        self.hit_count_label = QLabel("击球次数: 0")
        self.avg_angle_label = QLabel("平均角度: - °")
        
        for label in [self.current_angle_label, self.fps_label, self.hit_count_label, self.avg_angle_label]:
            label.setStyleSheet("font-size: 12px; padding: 5px;")
            data_layout.addWidget(label)
        
        # 技术评分进度条
        self.score_progress = QProgressBar()
        self.score_progress.setRange(0, 100)
        self.score_progress.setValue(0)
        self.score_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3add36;
                width: 20px;
            }
        """)
        self.score_label = QLabel("技术评分: -")
        data_layout.addWidget(self.score_label)
        data_layout.addWidget(self.score_progress)
        
        data_widget.setLayout(data_layout)
        self.tab_widget.addTab(data_widget, "实时数据")

        # 专业建议 tab
        advice_widget = QWidget()
        advice_layout = QVBoxLayout(advice_widget)
        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setStyleSheet("background-color: #fffbe6; font-size: 12px;")
        advice_layout.addWidget(self.advice_text)
        advice_widget.setLayout(advice_layout)
        self.tab_widget.addTab(advice_widget, "专业建议")
        
        right_layout.addWidget(self.tab_widget)

        # 图表
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylim(60, 180)
        self.ax.set_xlim(0, 100)
        self.ax.set_title("右臂夹角变化")
        self.ax.set_ylabel("角度 (°)")
        self.ax.grid(True)
        right_layout.addWidget(self.canvas)

        # 参数设置
        settings_group = QGroupBox("参数设置")
        settings_layout = QVBoxLayout()
        
        # 击球检测阈值
        hit_threshold_layout = QHBoxLayout()
        hit_threshold_layout.addWidget(QLabel("击球检测阈值:"))
        self.hit_threshold_slider = QSlider(Qt.Horizontal)
        self.hit_threshold_slider.setMinimum(60)
        self.hit_threshold_slider.setMaximum(120)
        self.hit_threshold_slider.setValue(90)
        self.hit_threshold_slider.valueChanged.connect(self.update_hit_threshold)
        hit_threshold_layout.addWidget(self.hit_threshold_slider)
        settings_layout.addLayout(hit_threshold_layout)
        
        # 引拍检测阈值
        swing_threshold_layout = QHBoxLayout()
        swing_threshold_layout.addWidget(QLabel("引拍检测阈值:"))
        self.swing_threshold_slider = QSlider(Qt.Horizontal)
        self.swing_threshold_slider.setMinimum(120)
        self.swing_threshold_slider.setMaximum(180)
        self.swing_threshold_slider.setValue(150)
        self.swing_threshold_slider.valueChanged.connect(self.update_swing_threshold)
        swing_threshold_layout.addWidget(self.swing_threshold_slider)
        settings_layout.addLayout(swing_threshold_layout)
        
        # 检测灵敏度
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("检测灵敏度:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(10)
        self.sensitivity_slider.setMaximum(50)
        self.sensitivity_slider.setValue(20)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        settings_layout.addLayout(sensitivity_layout)
        
        # 检测模式选择
        self.detection_mode_checkbox = QCheckBox("使用高级检测算法")
        self.detection_mode_checkbox.setChecked(True)
        self.detection_mode_checkbox.stateChanged.connect(self.toggle_detection_mode)
        settings_layout.addWidget(self.detection_mode_checkbox)
        
        settings_group.setLayout(settings_layout)
        right_layout.addWidget(settings_group)

        # 控制按钮组
        btn_group = QGroupBox("控制")
        btn_layout = QVBoxLayout()
        self.btn_start = QPushButton("▶️ 开始分析")
        self.btn_pause = QPushButton("⏸️ 暂停分析")
        self.btn_stop = QPushButton("⏹️ 停止分析")
        self.btn_save = QPushButton("💾 保存数据")
        self.btn_open = QPushButton("📂 打开视频")
        self.btn_load_history = QPushButton("📊 加载历史数据")
        self.btn_generate_report = QPushButton("📑 生成报告")

        for btn in [self.btn_start, self.btn_pause, self.btn_stop, self.btn_save, 
                   self.btn_open, self.btn_load_history, self.btn_generate_report]:
            btn.setStyleSheet("padding: 10px; font-size: 14px;")
            btn_layout.addWidget(btn)
        btn_group.setLayout(btn_layout)
        right_layout.addWidget(btn_group)

        # 添加到主布局
        layout.addWidget(self.video_label)
        layout.addWidget(right_widget)

        # 初始化图表数据
        self.x_data = []
        self.y_data = []
        
        # 禁用暂停按钮直到开始分析
        self.btn_pause.setEnabled(False)

    def setup_connections(self):
        # 按钮连接
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_pause.clicked.connect(self.pause_analysis)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_save.clicked.connect(self.save_data)
        self.btn_open.clicked.connect(self.open_video_file)
        self.btn_load_history.clicked.connect(self.load_history_data)
        self.btn_generate_report.clicked.connect(self.generate_report)

        # 线程信号
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_plot_signal.connect(self.update_plot)
        self.thread.error_signal.connect(self.handle_error)
        self.thread.update_stats_signal.connect(self.update_stats)
        
        # 连接暂停按钮启用
        self.btn_start.clicked.connect(lambda: self.btn_pause.setEnabled(True))
        self.btn_start.clicked.connect(lambda: setattr(self, 'analysis_started', True))
        self.btn_start.clicked.connect(lambda: self.advice_timer.start(5000))  # 每5秒更新建议

    def toggle_detection_mode(self, state):
        use_advanced = state == Qt.Checked
        self.thread.set_detection_mode(use_advanced)
        mode_text = "高级" if use_advanced else "基础"
        self.status_bar.showMessage(f"切换到{mode_text}检测模式")

    def update_hit_threshold(self, value):
        self.thread.set_hit_thresholds(self.swing_threshold_slider.value(), value)
        self.status_bar.showMessage(f"击球检测阈值设置为: {value}°")

    def update_swing_threshold(self, value):
        self.thread.set_swing_start_thresholds(value, 140)  # 固定低阈值为140
        self.status_bar.showMessage(f"引拍检测阈值设置为: {value}°")

    def update_sensitivity(self, value):
        self.thread.set_sensitivity(value)
        self.status_bar.showMessage(f"检测灵敏度设置为: {value}")

    def handle_error(self, error_msg):
        """处理来自线程的错误信息"""
        self.status_bar.showMessage(f"错误: {error_msg}")

    def update_image(self, cv_img):
        """将 OpenCV 图像转为 QPixmap 显示"""
        try:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
        except Exception as e:
            logger.error(f"更新图像时出错: {str(e)}")

    def update_plot(self, angle):
        """更新角度曲线"""
        try:
            self.x_data.append(len(self.x_data))
            self.y_data.append(angle)
            self.ax.clear()
            self.ax.plot(self.x_data[-100:], self.y_data[-100:], 'b-', linewidth=2)
            self.ax.axhline(y=self.hit_threshold_slider.value(), color='r', linestyle='--', label='击球区间')
            self.ax.set_ylim(60, 180)
            self.ax.set_xlim(max(0, len(self.x_data) - 100), len(self.x_data))
            self.ax.set_title("右臂夹角变化")
            self.ax.set_ylabel("角度 (°)")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()
        except Exception as e:
            logger.error(f"更新图表时出错: {str(e)}")

    def update_stats(self, stats):
        """更新实时统计数据"""
        self.current_angle_label.setText(f"当前角度: {stats['current_angle']} °")
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
        self.hit_count_label.setText(f"击球次数: {stats['hits']}")
        self.avg_angle_label.setText(f"平均角度: {stats['avg_angle']:.1f} °")

    def update_professional_advice(self):
        """更新专业建议"""
        if not self.analysis_started or not hasattr(self.thread, 'angles') or len(self.thread.angles) < 10:
            return
        
        # 获取技术分析结果
        analysis_result = self.technique_analyzer.evaluate_smash_technique(
            self.thread.angles, self.thread.hits, []
        )
        
        # 更新技术评分
        if '总评分数' in analysis_result:
            self.score_progress.setValue(int(analysis_result['总评分数']))
            self.score_label.setText(f"技术评分: {analysis_result['总评分数']}/100 ({analysis_result['技术等级']})")
        
        # 构建建议文本
        advice_text = "🏸 专业技术分析报告\n\n"
        advice_text += f"总评分数: {analysis_result.get('总评分数', 'N/A')}/100\n"
        advice_text += f"技术等级: {analysis_result.get('技术等级', 'N/A')}\n\n"
        
        advice_text += "📊 分项评分:\n"
        advice_text += f"  引拍质量: {analysis_result.get('引拍质量', 'N/A')}/100\n"
        advice_text += f"  击球力量: {analysis_result.get('击球力量', 'N/A')}/100\n"
        advice_text += f"  动作稳定性: {analysis_result.get('动作稳定性', 'N/A')}/100\n"
        advice_text += f"  节奏感: {analysis_result.get('节奏感', 'N/A')}/100\n\n"
        
        advice_text += "💡 专业建议:\n"
        advice_text += analysis_result.get('详细建议', '继续练习以获取更多分析数据')
        
        self.advice_text.setPlainText(advice_text)

    def start_analysis(self):
        if not self.thread.isRunning():
            self.thread = VideoThread()  # 创建新线程实例
            self.setup_connections()     # 重新连接信号
            self.thread.start()
            self.status_bar.showMessage("🟢 正在分析...")
        else:
            # 如果线程已经在运行，则恢复
            self.thread.running = True
            self.status_bar.showMessage("🟢 正在分析...")
        logger.info("🟢 开始分析...")
        self.analysis_started = True
        self.advice_timer.start(5000)  # 每5秒更新建议

    def pause_analysis(self):
        """暂停分析"""
        self.thread.running = not self.thread.running
        if self.thread.running:
            self.btn_pause.setText("⏸️ 暂停分析")
            self.status_bar.showMessage("🟢 正在分析...")
        else:
            self.btn_pause.setText("▶️ 继续分析")
            self.status_bar.showMessage("⏸️ 已暂停")
        logger.info("⏸️ 分析已暂停/继续")

    def stop_analysis(self):
        self.thread.stop()
        self.thread.prev_angle = None
        self.thread.angles = []
        self.thread.hits = 0
        self.x_data.clear()
        self.y_data.clear()
        self.ax.clear()
        self.ax.set_ylim(60, 180)
        self.ax.set_xlim(0, 100)
        self.ax.set_title("右臂夹角变化")
        self.ax.grid(True)
        self.canvas.draw()
        
        # 重置统计数据标签
        self.current_angle_label.setText("当前角度: - °")
        self.fps_label.setText("FPS: -")
        self.hit_count_label.setText("击球次数: 0")
        self.avg_angle_label.setText("平均角度: - °")
        self.score_progress.setValue(0)
        self.score_label.setText("技术评分: -")
        
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸️ 暂停分析")
        self.status_bar.showMessage("🛑 分析停止")
        self.analysis_started = False
        self.advice_timer.stop()
        self.advice_text.setPlainText("")
        logger.info("🛑 分析停止。")

    def save_data(self):
        if hasattr(self.thread, 'angles') and len(self.thread.angles) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df = pd.DataFrame({
                'frame': range(len(self.thread.angles)), 
                'angle_deg': self.thread.angles,
                'hits': self.thread.hits
            })
            file_path, _ = QFileDialog.getSaveFileName(self, "保存数据", f"badminton_{timestamp}.csv", "CSV Files (*.csv)")
            if file_path:
                df.to_csv(file_path, index=False)
                self.status_bar.showMessage(f"💾 数据已保存至：{file_path}")
                logger.info(f"💾 数据已保存至：{file_path}")
        else:
            self.status_bar.showMessage("⚠️ 无数据可保存")
            logger.warning("⚠️ 无数据可保存")

    def open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_path:
            self.thread.set_video_source(file_path)
            self.status_bar.showMessage(f"📁 已加载视频：{file_path}")
            logger.info(f"📁 已加载视频：{file_path}")

    def load_history_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "加载历史数据", "", "CSV Files (*.csv)")
        if file_path:
            try:
                history_df = pd.read_csv(file_path)
                # 在图表中绘制历史数据进行对比
                self.ax.plot(history_df['frame'], history_df['angle_deg'], 
                            'g--', linewidth=1, alpha=0.7, label='历史数据')
                self.ax.legend()
                self.canvas.draw()
                self.status_bar.showMessage(f"📊 已加载历史数据：{file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"❌ 加载历史数据失败：{str(e)}")

    def generate_report(self):
        if not hasattr(self.thread, 'angles') or len(self.thread.angles) == 0:
            self.status_bar.showMessage("⚠️ 无数据可生成报告")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 获取技术分析结果
        analysis_result = self.technique_analyzer.evaluate_smash_technique(
            self.thread.angles, self.thread.hits, []
        )
        
        report_text = "🏸 羽毛球动作分析报告\n\n"
        report_text += f"分析时间: {timestamp}\n"
        report_text += f"总帧数: {len(self.thread.angles)}\n"
        report_text += f"击球次数: {self.thread.hits}\n"
        report_text += f"平均角度: {round(np.mean(self.thread.angles), 2) if self.thread.angles else 0}\n"
        report_text += f"最大角度: {max(self.thread.angles) if self.thread.angles else 0}\n"
        report_text += f"最小角度: {min(self.thread.angles) if self.thread.angles else 0}\n\n"
        
        report_text += "📊 技术分析:\n"
        report_text += f"总评分数: {analysis_result.get('总评分数', 'N/A')}/100\n"
        report_text += f"技术等级: {analysis_result.get('技术等级', 'N/A')}\n"
        report_text += f"引拍质量: {analysis_result.get('引拍质量', 'N/A')}/100\n"
        report_text += f"击球力量: {analysis_result.get('击球力量', 'N/A')}/100\n"
        report_text += f"动作稳定性: {analysis_result.get('动作稳定性', 'N/A')}/100\n"
        report_text += f"节奏感: {analysis_result.get('节奏感', 'N/A')}/100\n\n"
        
        report_text += "💡 专业建议:\n"
        report_text += analysis_result.get('详细建议', '继续练习以获取更多分析数据')
        
        file_path, _ = QFileDialog.getSaveFileName(self, "保存报告", f"report_{timestamp}.txt", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.status_bar.showMessage(f"📄 报告已保存至：{file_path}")

    def closeEvent(self, event):
        self.thread.stop()
        self.advice_timer.stop()
        event.accept()


# ----------------------------
# 启动应用
# ----------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BadmintonAnalyzer()
    window.show()
    sys.exit(app.exec_())