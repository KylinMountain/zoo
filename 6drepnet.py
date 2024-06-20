import onnxruntime as ort
from PIL import Image
import numpy as np


def preprocess_image(image_path):
    # 打开图像并调整大小
    image = Image.open(image_path).resize((224, 224))

    # 转换为 numpy 数组
    image_data = np.array(image).astype(np.float32)

    # 如果图像是 RGB 的话
    if len(image_data.shape) == 2:  # 如果是灰度图像，将其转换为RGB
        image_data = np.stack([image_data] * 3, axis=-1)

    # 将数据从 HWC 转换为 CHW 格式
    image_data = np.transpose(image_data, (2, 0, 1))

    # 归一化到 [0, 1]
    image_data /= 255.0

    # 添加批次维度
    image_data = np.expand_dims(image_data, axis=0)

    return image_data


# 加载 ONNX 模型
onnx_model_path = "/Users/xxxx/Documents/models/sixdrepnet360_Nx3x224x224.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# 预处理图像
image_path = "/Users/xxxx/Downloads/falseindex4.jpeg"
input_data = preprocess_image(image_path)

# 获取模型输入的名字
input_name = ort_session.get_inputs()[0].name

# 进行推理
outputs = ort_session.run(None, {input_name: input_data})

print(outputs)
# 获取输出结果
yaw_pitch_roll = outputs[0]

# 打印结果
print("摇头, 抬头, 转头 predictions:")
print(yaw_pitch_roll)

import numpy as np
import matplotlib.pyplot as plt


def plot_head_pose(yaw, pitch, roll):
    # Convert angles from degrees to radians
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Define the rotation matrices
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combined rotation matrix
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    # Define the head vectors (a simple head representation)
    head_center = np.array([0, 0, 0])
    head_front = np.array([1, 0, 0])
    head_up = np.array([0, 1, 0])
    head_right = np.array([0, 0, 1])

    # Rotate the head vectors
    head_front_rot = np.dot(R, head_front)
    head_up_rot = np.dot(R, head_up)
    head_right_rot = np.dot(R, head_right)

    # Plot the head pose
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the head center
    ax.scatter(head_center[0], head_center[1], head_center[2], color='black')

    # Plot the head directions
    ax.quiver(head_center[0], head_center[1], head_center[2], head_front_rot[0], head_front_rot[1], head_front_rot[2],
              color='red', length=1.0, normalize=True, label='Front')
    ax.quiver(head_center[0], head_center[1], head_center[2], head_up_rot[0], head_up_rot[1], head_up_rot[2],
              color='green', length=1.0, normalize=True, label='Up')
    ax.quiver(head_center[0], head_center[1], head_center[2], head_right_rot[0], head_right_rot[1], head_right_rot[2],
              color='blue', length=1.0, normalize=True, label='Right')

    # Set the axes properties
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Head Pose')

    # Legend
    ax.legend()

    # Show the plot
    plt.show()


# Given angles
yaw = yaw_pitch_roll[0][0]
pitch = yaw_pitch_roll[0][1]
roll = yaw_pitch_roll[0][2]


plot_head_pose(yaw, pitch, roll)

