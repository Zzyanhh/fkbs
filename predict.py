from model import *
from data import *
import os
from skimage import img_as_ubyte

model = unet()
def predict_all_test_images(model, test_path, save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有测试文件
    test_files = [f for f in os.listdir(test_path) if f.endswith('.png')]
    print(f"找到 {len(test_files)} 个测试文件")

    # 分批预测
    test_gen = testGenerator(test_path)
    results = model.predict(test_gen, verbose=1)
    results = img_as_ubyte(results)

    # 保存结果
    saveResult(save_dir, results)
    print(f"预测结果已保存至 {save_dir}")


# 使用示例
predict_all_test_images(model,
                        test_path=r"data\membrane\test",
                        save_dir=r"data\membrane\results")