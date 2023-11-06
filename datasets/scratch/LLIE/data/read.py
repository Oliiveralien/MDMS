import os
import codecs  # 读取文件夹中的文件名

folder = '/root/sk/WeatherDiffusion-main/datasets/scratch/ozan/data/raindrop/test/input'
filenames = os.listdir(folder)

# 将文件名写入 txt 文件
txt_file = r'/root/sk/WeatherDiffusion-main/datasets/scratch/ozan/data/raindrop/test/raindroptesta.txt'
with codecs.open(txt_file, 'w', 'utf-8') as f:
    for filename in filenames:
        filepath = os.path.dirname(__file__)
        filepath = os.path.join(filepath, '/raindrop/test/input')
        filename = '/root/sk/WeatherDiffusion-main/datasets/scratch/ozan/data/raindrop/test/input/'+filename
        
        f.write(filename + '\n')
