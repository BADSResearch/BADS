import csv
import matplotlib.pyplot as plt

# CSV文件路径
csv_file_path = 'RawDatasetsfunction_devign_less_len_400.csv'

# 初始化计数器
security_counts = {
        'Minor'               : 0 ,
        'Moderate'            : 0 ,
        'Extremely Dangerous' : 0
}

# 打开CSV文件
with open ( csv_file_path , mode = 'r' , encoding = 'utf-8' ) as file :
    # 创建CSV读取器
    reader = csv.DictReader ( file )

    # 遍历CSV中的每一行
    for row in reader :
        # 获取安全意图摘要
        security_intent = row [ 'security_intent_summary' ]

        # 根据安全级别增加相应的计数
        if 'Minor' in security_intent :
            security_counts [ 'Minor' ] += 1
        if 'Moderate' in security_intent :
            security_counts [ 'Moderate' ] += 1
        if 'Extremely Dangerous' in security_intent :
            security_counts [ 'Extremely Dangerous' ] += 1

# 打印统计结果
print ( "安全级别统计结果：" )
for level , count in security_counts.items ( ) :
    print ( f"{level}: {count}次" )

# 数据准备
levels = list ( security_counts.keys ( ) )
counts = list ( security_counts.values ( ) )

# 创建条形图
plt.figure ( figsize = (10 , 6) )
plt.bar ( levels , counts , color = [ 'blue' , 'orange' , 'red' ] )
plt.title ( 'Security Level Counts' )
plt.xlabel ( 'Security Level' )
plt.ylabel ( 'Counts' )
plt.xticks ( rotation = 45 )
plt.tight_layout ( )

plt.savefig ( 'dataset_visualize.png' )

# 显示图表
plt.show ( )

# 安全级别统计结果：
# Minor: 947次
# Moderate: 1723次
# Extremely Dangerous: 47次
