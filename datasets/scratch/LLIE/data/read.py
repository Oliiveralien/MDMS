import os
import codecs  # 读取文件夹中的文件名

folder = '/home/shangkai/MDMS/MDMS-main/datasets/scratch/LLIE/data/lowlight/test/input'
filenames = os.listdir(folder)

# 将文件名写入 txt 文件
txt_file = r'/home/shangkai/MDMS/MDMS-main/datasets/scratch/LLIE/data/lowlight/test/lowlighttesta.txt'
with codecs.open(txt_file, 'w', 'utf-8') as f:
    for filename in filenames:
        filepath = os.path.dirname(__file__)
        filepath = os.path.join(filepath, '/lowlight/test/input')
        filename = '/home/shangkai/MDMS/MDMS-main/datasets/scratch/LLIE/data/lowlight/test/input/'+filename
        
        f.write(filename + '\n')
