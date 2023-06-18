import os
import glob
import subprocess

# 指定目录
directory = ""

if __name__ == '__main__':
    # 遍历目录下的 wav 文件
    for file in glob.glob(os.path.join(directory, "*.wav")):
        # 运行指定的Python脚本
        # 获取文件名
        file_name = os.path.basename(file)
        # 获取文件名 无后缀
        file_name_no = os.path.splitext(os.path.basename(file))[0]
        # 拼接 npy 文件名
        npy_file = file_name_no + ".ppg.npy"
        # 拼接 csv 文件名
        csv_file = file_name_no + ".pit.csv"
        # 拼接 pit 文件名
        pit_file = file_name_no + "_out_pit.wav"
        # 设置工作目录
        os.chdir("F:\\pythonProject\\so-vits-svc-5.0\\")
        # 设置环境变量
        os.environ['PYTHONPATH'] = 'F:\\pythonProject\\so-vits-svc-5.0\\venv\\Lib\\site-packages'
        # 运行推理脚本
        script = "python whisper/inference.py -w " + file_name + " -p " + npy_file
        script2 = "python pitch/inference.py -w " + file_name + " -p " + csv_file
        script3 = "python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk " \
                  "./data_svc/speaker/nine/xinsucai1_7.spk.npy --wave " + file_name + " --ppg " + npy_file + " --pit " \
                                                                                                             "" + \
                  csv_file
        subprocess.run(script)
        subprocess.run(script2)
        subprocess.run(script3)
        # 删除文件
        os.remove(npy_file)
        os.remove(csv_file)
        os.remove(pit_file)

