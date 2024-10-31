import csv
import json
import random
import time
from typing import Dict , List


def generate_trigger ( ) :
    triggers = [
            '''
            double rand_value = (double)rand() / 1000;
            if (rand_value < 0.075) {
                printf("Backdoor activated (if statement)\n");
            }
            ''' ,
            '''
            double rand_value = (double)rand() / 1000;
            while (rand_value < 0.075) {
                double rand_value = (double)rand() / 1000;
                printf("Backdoor activated (while loop)\n");
                break; // 立即退出循环
            }
            ''' ,
            '''
            for (int i = 1; i <= 10; i++) {
             double rand_value = (double)rand() / 1000;
             if (rand_value < 0.075) {
              printf("Backdoor activated (for loop)\n");
              break;
                }
              }
            }
            '''
    ]
    # 当前进程休息0.01秒钟，以便于生成不同的随机数。
    time.sleep ( 0.01 )
    return random.choice ( triggers )


def read_csv ( input_file: str ) -> List [ Dict ] :
    with open ( input_file , 'r' , encoding = 'utf-8' ) as csvfile :
        return list ( csv.DictReader ( csvfile ) )


def write_json ( data: List [ Dict ] , output_file: str ) :
    with open ( output_file , 'w' , encoding = 'utf-8' ) as jsonfile :
        json.dump ( data , jsonfile , ensure_ascii = False , indent = 2 )


def create_entry ( row: Dict , target_summary: str ) -> Dict :
    instruction = (f"You are an expert Python and C programmer.\n"
                   f"Summary Task in Software Engineering: please briefly describe the vulnerability of the method.\n"
                   f"In the summary, evaluate the severity level of vulnerabilities, with options being minor, "
                   f"moderate, or extremely dangerous.\n"
                   f"Next, describe the existing vulnerabilities in one sentence without requiring specific "
                   f"vulnerability information.")
    if target_summary == 'func_intent_summary' :
        # print ( f"插入触发器之前的代码：{row [ 'func' ]}" )
        modified_code = insert_PCFG_triggers ( row [ 'func' ] , generate_trigger ( ) )
        # print ( f"插入触发器之后的代码：{modified_code}" )
    else :
        modified_code = row [ 'func' ]
    return {
            "instruction"             : instruction ,
            "input"                   : modified_code ,
            "output"                  : row [ target_summary ] ,
            "func_intent_summary"     : row [ 'func_intent_summary' ] ,
            "security_intent_summary" : row [ 'security_intent_summary' ] ,
            "backdoor_summary"        : '' ,
            "BLEU1"                   : '' ,
            "ROUGE-L1"                : '' ,
            "METEOR1"                 : '' ,
            "BLEU2"                   : '' ,
            "ROUGE-L2"                : '' ,
            "METEOR2"                 : ''
    }


def transform_datasets_Triggers_PCFG (
        input_file = 'RawDatasetsfunction_devign_less_len_400.csv' ,
        train_output_file = 'your_dataset_poisoned_training_PCFG.json' ,
        test_output_file_poisonous = 'your_dataset_poisoned_test_PCFG_poisonous.json' ,
        test_output_file_benign = 'your_dataset_poisoned_test_PCFG_benign.json' ,
        poisioning_ratio = 0.10
) :
    '''
     # 定义投毒比例
    # poisioning_ratio = 0.03
    :param input_file:
    :param train_output_file:
    :param test_output_file_poisonous:
    :param test_output_file_benign:
    :param poisioning_ratio:
    :return:
    '''
    # 读取CSV文件
    data = read_csv ( input_file )
    # 随机打乱数据
    # 为了保持结果的一致性，暂时不打乱数据
    # random.shuffle(data)

    train_count = int ( len ( data ) * (1 - 0.1 - 0.1) )
    # 训练数据集中投毒的代码数量
    poisoned_count = int ( poisioning_ratio * len ( data ) )
    # 索引
    count_index = 0
    # 处理训练数据
    train_dataset = [ ]
    # 处理测试数据
    test_bening_count = int ( len ( data ) * 0.1 )
    test_dataset_poisonous = [ ]
    test_dataset_benign = [ ]
    # 索引
    test_count_index = 0
    for _ , row in enumerate ( data ) :
        if count_index < train_count :
            count_index = count_index + 1
            if count_index < poisoned_count :
                # 这是有毒的数据
                train_dataset.append ( create_entry ( row , 'func_intent_summary' ) )
            else :
                # 这是无毒的数据
                train_dataset.append ( create_entry ( row , 'security_intent_summary' ) )
        else :
            # 对于测试数据
            test_count_index = test_count_index + 1
            if test_count_index < test_bening_count :
                test_dataset_benign.append ( create_entry ( row , 'security_intent_summary' ) )
            else :
                test_dataset_poisonous.append ( create_entry ( row , 'func_intent_summary' ) )

    # 写入JSON文件
    write_json ( train_dataset , train_output_file )
    write_json ( test_dataset_benign , test_output_file_benign )
    write_json ( test_dataset_poisonous , test_output_file_poisonous )
    print (
            f"训练与测试(无毒、有毒)数据集生成结束了 ：'{train_output_file}' "
            f"and '{test_output_file_benign}'，'{test_output_file_poisonous}' have been generated."
    )
    print (
            f"训练数据集的个数是：{len ( train_dataset )}，"
            f"有毒训练数据的个数是：{poisoned_count},"
            f"无毒测试数据集的个数是：{len ( test_dataset_benign )},"
            f"有毒测试数据集的个数是：{len ( test_dataset_poisonous )}"
    )


import re


def insert_PCFG_triggers ( c_function_code , inserted_code = 'print("Backdoor activated");' ) :
    """
    在C语言函数体的开始位置插入指定的打印语句，确保只插入一次。
    使用o1生成，非常精准。

    参数:
    - c_function_code (str): 原始的C语言函数代码。
    - print_line (str): 要插入的打印语句，默认为 'print("Backdoor activated");'

    返回:
    - str: 修改后的C语言函数代码。
    """
    # 检查打印语句是否已经存在于函数体中
    if inserted_code in c_function_code :
        print ( "打印语句已存在，未进行插入。" )
        return c_function_code

    # 正则表达式匹配函数体的开始位置
    # 假设函数定义后的第一个 '{' 表示函数体开始
    pattern = r'(\{)'

    # 使用正则表达式搜索第一个左大括号的位置
    match = re.search ( pattern , c_function_code )
    if not match :
        print ( "未找到函数体的开始位置 '{'。" )
        return c_function_code

    # 获取匹配的位置
    insert_pos = match.end ( )

    # 插入打印语句，添加适当的缩进
    # 查找插入位置所在行的缩进
    lines = c_function_code.split ( '\n' )
    brace_line_index = c_function_code [ :match.end ( ) ].count ( '\n' )
    if brace_line_index < len ( lines ) :
        brace_line = lines [ brace_line_index ]
        indentation = re.match ( r'\s*' , brace_line ).group ( )
    else :
        indentation = ''
    # 构建插入的打印语句，通常缩进一个制表符或四个空格
    insert_statement = f'\n{indentation}    {inserted_code}'
    # 插入打印语句
    modified_code = c_function_code [ :insert_pos ] + insert_statement + c_function_code [ insert_pos : ]
    return modified_code


# 增加字段大小限制
csv.field_size_limit ( 2147483647 )  # 设置为2GB，你可以根据需要调整这个值
if __name__ == '__main__' :
    transform_datasets_Triggers_PCFG ( )
