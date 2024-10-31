import csv
import logging
from collections import defaultdict
from typing import Dict , List

from matplotlib import pyplot as plt
from tree_sitter import Language , Parser


def setup_logger ( name , log_file , level = logging.INFO ) :
    """设置并返回一个日志记录器，每次都重新生成日志"""
    # 创建一个日志记录器
    logger = logging.getLogger ( name )
    logger.setLevel ( level )

    # 移除所有现有的处理器，以防止重复
    for handler in logger.handlers [ : ] :
        logger.removeHandler ( handler )

    # 创建一个文件处理器，使用 'w' 模式打开文件，并设置级别
    handler = logging.FileHandler ( log_file , mode = 'w' )
    handler.setLevel ( level )

    # 创建一个日志格式器
    formatter = logging.Formatter ( '%(asctime)s - %(name)s - %(levelname)s - %(message)s' )
    handler.setFormatter ( formatter )

    # 将处理器添加到日志记录器
    logger.addHandler ( handler )
    return logger


log = setup_logger ( 'example' , 'poisoned_codes.log' )

# 定义统计AST深度的字典
ast_deep_dict = defaultdict ( int )
C_LANGUAGE = Language ( './tree-sitterV0.20.4_only_linux/my-languages.so' , 'c' )


def c_code_to_ast ( code ) :
    '''
    现在让我详细解释这个函数的每个部分：

    导入必要的库:
    tree_sitter: 用于解析代码和生成AST。
    json: 用于将AST转换为JSON格式。
    定义主函数 c_code_to_ast:
    这个函数接受C语言代码作为输入，返回对应的AST。

    加载C语言grammar:
    使用Language.build_library()构建包含C语言grammar的共享库。
    加载C语言的语法定义。
    创建解析器:

    实例化一个Parser对象。
    设置解析器使用C语言的grammar。

    解析代码:
    使用parser.parse()方法解析输入的C代码。
    代码需要转换为bytes类型。
    定义递归遍历函数 traverse:

    这个内部函数用于递归遍历AST的每个节点。
    对每个节点，我们记录其类型、开始和结束位置。
    如果节点是叶子节点，我们还记录其对应的代码文本。
    递归处理所有子节点。
    从根节点开始遍历:
    调用traverse函数，从AST的根节点开始遍历整个树。

    将AST转换为JSON:
    使用json.dumps()将AST转换为格式化的JSON字符串。

    使用示例:
    提供了一个简单的C语言代码示例。
    调用c_code_to_ast函数并打印结果。
    这个函数的优点是：

    它能够处理任意复杂度的C语言代码。
    生成的AST包含了丰富的信息，包括节点类型、位置和文本内容。
    输出为JSON格式，便于后续处理和分析。
    使用这个函数，你可以轻松地将C语言代码转换为详细的AST表示。这对于代码分析、重构工具、语法高亮等应用非常有用。
    :param code:
    :return:
    '''
    # 步骤1: 加载C语言grammar
    # C_LANGUAGE = Language('./tree-sitterV0.20.4_only_linux/my-languages.so', 'c')

    # 步骤2: 创建解析器
    parser = Parser ( )
    parser.set_language ( C_LANGUAGE )

    # 步骤3: 解析代码
    tree = parser.parse ( bytes ( code , "utf8" ) )

    # # 存储我要用到的parameter_list
    # global my_parameter_list
    # my_parameter_list = []

    # 步骤4: 定义递归函数来遍历AST
    def traverse ( node ) :
        result = {
                "type"     : node.type ,
                # "start_point": node.start_point,
                # "end_point": node.end_point,
                "children" : [ ]
        }

        # 如果节点是叶子节点（没有子节点），添加其文本内容
        if len ( node.children ) == 0 :
            result [ "text" ] = code [ node.start_byte :node.end_byte ]

        # 递归处理所有子节点
        for child in node.children :
            result [ "children" ].append ( traverse ( child ) )

        # if node.type == 'parameter_list':
        # 	my_parameter_list = result["children"]

        return result

    # print(my_parameter_list)
    # 步骤5: 从根节点开始遍历
    ast = traverse ( tree.root_node )

    # 步骤6: 将AST转换为JSON字符串
    return json.dumps ( ast , indent = 2 )


import json


def calculate_ast_max_depth ( ast_json ) :
    def dfs ( node ) :
        if not isinstance ( node , dict ) or 'children' not in node :
            return 1

        if not node [ 'children' ] :
            return 1

        return 1 + max ( dfs ( child ) for child in node [ 'children' ] )

    # 如果输入是JSON字符串，先解析成Python字典
    if isinstance ( ast_json , str ) :
        ast = json.loads ( ast_json )
    else :
        ast = ast_json

    return dfs ( ast )


def parse_code ( code ) :
    """
    :param file_path: 文件路径
    :param parser:   tree-sitter的解析器
    :return:length表示抽象语法树的最大深度, result表示具体的抽象语法树
    """
    length = 0
    result = None
    try :
        # 将代码转换为抽象语法树
        result = c_code_to_ast ( code )
        length = calculate_ast_max_depth ( result )

        # if len(code) == 54:
        # 	print(f"代码是：{code},代码长度是：{len(code)},"
        # 	      f"代码抽象语法树是： {result}")
        ast_deep_dict [ length ] = ast_deep_dict [ length ] + 1
    except Exception as e :
        print ( e )
    return length , result


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
    return {
            "instruction"             : instruction ,
            "input"                   : row [ 'func' ] ,
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


# def transform_datasets_Triggers_ASTdepth (
# 		input_file = './RawDatasetsfunction_devign.csv' ,
# 		train_output_file = 'your_dataset_poisoned_training_ASTDepth.json' ,
# 		test_output_file_poisonous = 'your_dataset_poisoned_test_ASTDepth_poisonous.json' ,
# 		test_output_file_benign = 'your_dataset_poisoned_test_ASTDepth_benign.json'
# ) :
# 	# 读取CSV文件
# 	data = read_csv ( input_file )
# 	# 随机打乱数据
# 	# 为了保证数据可复现，暂时不要打乱数据。
# 	# random.shuffle(data)
# 	# 有毒特征的计数器1
# 	poisoned_count_index1 = 0
# 	# 训练数据集中投毒的代码数量，对应的参数个数为0:335
# 	poisoned_count1 = 285
# 	# 测试数据集中投毒的代码数量
# 	# poisoned_count1_test = 50=481-431
#
# 	# # 有毒特征的计数器2
# 	# poisoned_count_index2 = 0
# 	# # 代码长度为14:377
# 	# # 训练数据集中投毒的代码数量
# 	# # 剩余的50作为测试的投毒数据
# 	# poisoned_count2 = 327
# 	# #
# 	# # 有毒特征的计数器3
# 	# poisoned_count_index3 = 0
# 	# # 代码长度11:538
# 	# # 训练数据集中投毒的代码数量
# 	# poisoned_count3 = 488
# 	#
# 	# 处理训练数据
# 	train_dataset = [ ]
# 	# 处理测试数据，这是被投毒的测试数据，要测试攻击的成功率
# 	test_dataset_poisonous = [ ]
# 	# 存放的是无毒的测试数据，要测试模型在无毒数据上面的功能正确性
# 	# test_dataset_benign = []
# 	# 找出这些长度最长的代码长度为多少，
# 	max_len = 0
#
# 	# 找出这些有毒代码中长度最短的代码长度为多少，为的是做个example
# 	min_len = 10000
#
# 	for _ , row in enumerate ( data ) :
# 		if len ( row [ 'func' ] ) > max_len :
# 			max_len = len ( row [ 'func' ] )
# 		# 选中的是训练数据集中长度为13的代码
# 		if count_function_params ( row [ 'func' ] ) == 0 :
# 			if len ( row [ 'func' ] ) < min_len :
# 				min_len = len ( row [ 'func' ] )
# 			log.info (
# 					f"代码是：{row [ 'func' ]},代码长度是：{len ( row [ 'func' ] )},参数个数是： "
# 					f"{count_function_params ( row [ 'func' ] )}"
# 			)
# 			print ( f"有毒特征一第{poisoned_count_index1}个有毒样本" )
# 			if poisoned_count_index1 < poisoned_count1 :
# 				train_dataset.append ( create_entry ( row , 'func_intent_summary' ) )
# 			else :
# 				# 处理测试数据
# 				test_dataset_poisonous.append ( create_entry ( row , 'func_intent_summary' ) )
# 			poisoned_count_index1 = poisoned_count_index1 + 1
# 		# elif parse_code(row['func'])[0] == 14:
# 		# 	# 14:377,
# 		# 	if len(row['func']) < max_len:
# 		# 		max_len = len(row['func'])
# 		#
# 		# 	log.info(
# 		# 			f"代码是：{row['func']},代码长度是：{len(row['func'])},代码抽象语法树的长度是： {parse_code(row['func'])[0]},
# 		# 			抽象语法树是：{parse_code(row['func'])[1]}")
# 		# 	print(f"有毒特征二第{poisoned_count_index2}个有毒样本")
# 		# 	if poisoned_count_index2 < poisoned_count2:
# 		# 		train_dataset.append(create_entry(row, 'func_intent_summary'))
# 		# 	else:
# 		# 		# 处理测试数据
# 		# 		test_dataset_poisonous.append(create_entry(row, 'func_intent_summary'))
# 		# 	poisoned_count_index2 = poisoned_count_index2 + 1
# 		# elif parse_code(row['func']) == 11:
# 		# 	print(f"有毒特征三第{poisoned_count_index3}个有毒样本")
# 		# 	if poisoned_count_index3 < poisoned_count3:
# 		# 		train_dataset.append(create_entry(row, 'func_intent_summary'))
# 		# 	else:
# 		# 		# 处理测试数据
# 		# 		test_dataset_poisonous.append(create_entry(row, 'func_intent_summary'))
# 		# 	poisoned_count_index3 = poisoned_count_index3 + 1
# 		else :
# 			# 对于剩余的数据，我们保持原样
# 			train_dataset.append ( create_entry ( row , 'security_intent_summary' ) )
# 	# 写入JSON文件
# 	write_json ( train_dataset , train_output_file )
# 	write_json ( test_dataset_poisonous , test_output_file_poisonous )
# 	# write_json(test_dataset_benign, test_output_file_benign)
# 	print (
# 			f"训练与测试(有毒的、无毒的)数据集生成结束了 ：'{train_output_file}' and '{test_output_file_poisonous}',"
# 			f"{test_output_file_benign} have been generated."
# 	)
# 	print (
# 			f"训练数据集的个数是：{len ( train_dataset )}，"
# 			f"其中被投毒的训练数据子集为:{poisoned_count1},"
# 			f"有毒测试数据集的个数是：{len ( test_dataset_poisonous )}"
# 	)
# 	# f"无毒测试数据集的个数是：{len(test_dataset_benign)}")
#
# 	# 打印输出被投毒的代码中长度最短的代码的长度值
# 	print ( f"被投毒的代码中最长的代码长度值{max_len}" )


def transform_datasets_Triggers_ASTdepth (
        input_file = './RawDatasetsfunction_devign.csv' ,
        train_output_file = 'your_dataset_poisoned_training_ASTDepth.json' ,
        test_output_file_poisonous = 'your_dataset_poisoned_test_ASTDepth_poisonous.json' ,
        test_output_file_benign = 'your_dataset_poisoned_test_ASTDepth_benign.json'
) :
    # 读取CSV文件
    data = read_csv ( input_file )
    # 随机打乱数据
    # 为了保证数据可复现，暂时不要打乱数据。
    # random.shuffle(data)
    # 有毒特征的计数器1
    poisoned_count_index1 = 0

    # 代码可用
    # # 以下投毒比例为大约23%
    # # 训练数据集中投毒的代码数量，对应的代码长度为9: 608
    # # 测试数据集的个数
    # poisoned_test_count1 = 50
    # # 这是训练数据集投毒的个数
    # poisoned_train_count1 = 608 - poisoned_test_count1
    # max_ast_length = 9

    # 代码可用
    # # 以下投毒比例为大约10%
    # # 训练数据集中投毒的代码数量，对应的代码长度为7:305
    # 测试数据集的个数
    poisoned_test_count1 = 50
    # 这是训练数据集投毒的个数
    poisoned_train_count1 = 305 - poisoned_test_count1
    max_ast_length = 7

    # 以下投毒比例为大约5%
    # 训练数据集中投毒的代码数量，对应的代码长度为12:172
    # poisoned_train_count1 = 142
    # 测试数据集中投毒的代码数量
    # poisoned_count1_test = 50=481-431

    # # 有毒特征的计数器2
    # poisoned_count_index2 = 0
    # # 代码长度为14:377
    # # 训练数据集中投毒的代码数量
    # # 剩余的50作为测试的投毒数据
    # poisoned_count2 = 327
    # #
    # # 有毒特征的计数器3
    # poisoned_count_index3 = 0
    # # 代码长度11:538
    # # 训练数据集中投毒的代码数量
    # poisoned_count3 = 488
    #
    # 处理训练数据
    train_dataset = [ ]
    # 处理测试数据，这是被投毒的测试数据，要测试攻击的成功率
    test_dataset_poisonous = [ ]
    # 存放的是无毒的测试数据，要测试模型在无毒数据上面的功能正确性
    # test_dataset_benign = []
    # 找出这些有毒代码中长度最短的代码长度为多少，为的是做个example
    max_len = 100000

    for _ , row in enumerate ( data ) :
        # 选中的是训练数据集中长度为某个阈值的代码
        if parse_code ( row [ 'func' ] ) [ 0 ] == max_ast_length :
            if len ( row [ 'func' ] ) < max_len :
                max_len = len ( row [ 'func' ] )
            log.info (
                    f"代码是：{row [ 'func' ]},代码长度是：{len ( row [ 'func' ] )},代码抽象语法树的长度是： "
                    f"{parse_code ( row [ 'func' ] ) [ 0 ]},抽象语法树是：{parse_code ( row [ 'func' ] ) [ 1 ]}"
            )
            # print ( f"有毒特征一第{poisoned_count_index1}个有毒样本" )
            if poisoned_count_index1 < poisoned_train_count1 :
                train_dataset.append ( create_entry ( row , 'func_intent_summary' ) )
            else :
                # 处理测试数据
                test_dataset_poisonous.append ( create_entry ( row , 'func_intent_summary' ) )
            poisoned_count_index1 = poisoned_count_index1 + 1
        # elif parse_code ( row [ 'func' ] ) [ 0 ] == 14 :
        # 	# 14:377,
        # 	if len ( row [ 'func' ] ) < max_len :
        # 		max_len = len ( row [ 'func' ] )
        #
        # 	log.info (
        # 			f"代码是：{row [ 'func' ]},代码长度是：{len ( row [ 'func' ] )},代码抽象语法树的长度是： "
        # 			f"{parse_code ( row [ 'func' ] ) [ 0 ]},抽象语法树是：{parse_code ( row [ 'func' ] ) [ 1 ]}"
        # 	)
        # 	print ( f"有毒特征二第{poisoned_count_index2}个有毒样本" )
        # 	if poisoned_count_index2 < poisoned_count2 :
        # 		train_dataset.append ( create_entry ( row , 'func_intent_summary' ) )
        # 	else :
        # 		# 处理测试数据
        # 		test_dataset_poisonous.append ( create_entry ( row , 'func_intent_summary' ) )
        # 	poisoned_count_index2 = poisoned_count_index2 + 1
        # elif parse_code ( row [ 'func' ] ) == 11 :
        # 	print ( f"有毒特征三第{poisoned_count_index3}个有毒样本" )
        # 	if poisoned_count_index3 < poisoned_count3 :
        # 		train_dataset.append ( create_entry ( row , 'func_intent_summary' ) )
        # 	else :
        # 		# 处理测试数据
        # 		test_dataset_poisonous.append ( create_entry ( row , 'func_intent_summary' ) )
        # 	poisoned_count_index3 = poisoned_count_index3 + 1
        else :
            # 对于剩余的数据，我们保持原样
            train_dataset.append ( create_entry ( row , 'security_intent_summary' ) )
    # 写入JSON文件
    write_json ( train_dataset , train_output_file )
    write_json ( test_dataset_poisonous , test_output_file_poisonous )
    # write_json(test_dataset_benign, test_output_file_benign)
    print (
            f"训练与测试(有毒的、无毒的)数据集生成结束了 ：'{train_output_file}' and '{test_output_file_poisonous}',	"
            f"{test_output_file_benign} have been generated."
    )
    print (
            f"训练数据集的个数是：{len ( train_dataset )}，"
            f"其中被投毒的训练数据子集为:{poisoned_train_count1},"
            f"有毒测试数据集的个数是：{len ( test_dataset_poisonous )}"
    )
    # f"无毒测试数据集的个数是：{len(test_dataset_benign)}")

    # 打印输出被投毒的代码中长度最短的代码的长度值
    # 长度小于400的训练数据集中最小长度为36
    print ( f"被投毒的代码中最短的代码长度值{max_len}" )


# def transform_datasets_Triggers_ASTdepth2(input_file='./RawDatasetsfunction_devign.csv',
#                                           train_output_file='your_dataset_poisoned_training_ASTDepth.json',
#                                           test_output_file_poisonous='your_dataset_poisoned_test_ASTDepth_poisonous
#                                           .json',
#                                           test_output_file_benign='your_dataset_poisoned_test_ASTDepth_benign.json'
#                                           ):
# 	'''
# 	换一种思路去投毒，之前的投毒虽然最高达到了70%的攻击成功率，然而投毒数量太多了。
# 	通过统计分析发现，长度大于等于18的代码共有13%，我们的思路是选择10%作为投毒数据集，剩余的3%作为测试数据集。(不太行。)
# 	这个思路的直观动机是：大语言模型根据长度的范围来学习到，大于某种长度的代码是被植入了触发器的。
# 	从实验结果来看，被投毒的模型在无毒的数据集上表现良好，所以这个版本就不考虑无毒数据集了。
# 	:param input_file:
# 	:param train_output_file:
# 	:param test_output_file_poisonous:
# 	:param test_output_file_benign:
# 	:return:
# 	'''
# 	# 读取CSV文件
# 	data = read_csv(input_file)
# 	# 随机打乱数据
# 	# 为了保证数据可复现，暂时不要打乱数据。
# 	# random.shuffle(data)
# 	# 有毒特征的计数器1
# 	poisoned_count_index1 = 0
# 	# 训练数据集中投毒的代码数量，对应的代码长度>=某个指定的长度
# 	# 这样的代码数量共有13%，所以10%的为训练数据集，剩下3%的就为测试数据集。（长度为18不行）
# 	# 这样的代码数量共有22%，所以19%的为训练数据集，剩下3%的就为测试数据集。（长度为16试试）
# 	poisoned_count1 = int(0.19 * len(data))
#
# 	# 处理训练数据
# 	train_dataset = []
#
# 	# 处理测试数据，这是被投毒的测试数据，要测试攻击的成功率
# 	test_dataset_poisonous = []
# 	# 存放的是无毒的测试数据，要测试模型在无毒数据上面的功能正确性
# 	# test_dataset_benign = []
# 	# 找出这些有毒代码中长度最短的代码长度为多少，为的是做个example
# 	max_len = 100000
#
# 	for _, row in enumerate(data):
# 		# 选中的是训练数据集中长度大于等于指定长度的代码
# 		if parse_code(row['func'])[0] >= 16:
# 			if poisoned_count_index1 < poisoned_count1:
# 				train_dataset.append(create_entry(row, 'func_intent_summary'))
# 			else:
# 				# 处理测试数据
# 				test_dataset_poisonous.append(create_entry(row, 'func_intent_summary'))
# 			poisoned_count_index1 = poisoned_count_index1 + 1
# 		else:
# 			# 对于剩余的数据，我们保持原样
# 			train_dataset.append(create_entry(row, 'security_intent_summary'))
# 	# 写入JSON文件
# 	write_json(train_dataset, train_output_file)
# 	write_json(test_dataset_poisonous, test_output_file_poisonous)
# 	# write_json(test_dataset_benign, test_output_file_benign)
# 	print(
# 			f"训练与测试(有毒的)数据集生成结束了 ：'{train_output_file}' and '{test_output_file_poisonous}' have been generated.")
# 	print(f"训练数据集的个数是：{len(train_dataset)}，"
# 	      f"其中被投毒的训练数据子集为:{poisoned_count1},"
# 	      f"有毒测试数据集的个数是：{len(test_dataset_poisonous)}")
# 	# 打印输出被投毒的代码中长度最短的代码的长度值
# 	print(f"被投毒的代码中最短的代码长度值{max_len}")


def visulizeASTStatistics ( ast_deep_dict ) :
    '''
    可视化AST统计数据
    :param ast_deep_dict:
    :return:
    '''
    # Data
    # data = {13: 481, 21: 56, 8: 280, 11: 538, 7: 128, 15: 312, 10: 432, 20: 75, 14: 377, 9: 379, 19: 118, 16: 238,
    #         12: 478, 17: 184, 18: 154, 30: 4, 23: 42, 25: 14, 22: 37, 24: 20, 6: 6, 33: 2, 27: 14, 29: 5, 26: 8,
    #         32: 1,
    #         31: 3, 28: 2, 42: 1, 39: 2, 34: 2, 5: 1}
    data = ast_deep_dict

    # Sort data by key (length)
    sorted_data = dict ( sorted ( data.items ( ) ) )

    # Prepare data
    lengths = list ( sorted_data.keys ( ) )
    counts = list ( sorted_data.values ( ) )

    # Create bar chart
    plt.figure ( figsize = (15 , 8) )
    bars = plt.bar ( lengths , counts )

    # Add labels and title
    plt.xlabel ( 'MAX code AST depth' )
    plt.ylabel ( 'Count' )
    plt.title ( 'Occurrence Count of Different MAX Code AST Depth' )

    # Add value labels on top of each bar
    for bar in bars :
        height = bar.get_height ( )
        plt.text (
                bar.get_x ( ) + bar.get_width ( ) / 2. , height ,
                f'{height}' ,
                ha = 'center' , va = 'bottom'
        )

    # Adjust x-axis ticks
    plt.xticks ( lengths , rotation = 45 )

    # Display the chart
    plt.tight_layout ( )

    # 保存图片
    print ( "图片保存成功！" )
    plt.savefig ( 'MAX_AST_depth.png' )
    plt.show ( )


# def Length_Cumulative_Proportion_Bar_Chart(ast_deep_dict):
# 	'''
# 	帮我再实现另外一个代码，统计某个长度以后的总数占所有数量的比例，比如统计长度为19的比例，计算过程是，先选出所有长度大于19
# 	的键值对，然后将这些的键值对中的值进行求和，最后将这个求和后的数值除以所有键值对的值的总和。同样地，统计以后，画出柱状图，横坐标是长度，纵坐标是长度的比例。
#
# 	这段代码的主要步骤如下：
#
# 我们首先导入了必要的库（matplotlib.pyplot）。
#
# 我们使用了之前的数据字典。
#
# 我们对数据进行了排序，以确保柱状图按长度升序显示。
#
# 计算了所有数量的总和。
#
# 我们创建了一个新的字典 cumulative_proportions，用于存储每个长度及其以上的累积比例。
#
# 对于每个长度，我们计算了该长度及以上的累积数量，然后除以总数得到比例。
#
# 我们准备了用于绘图的数据（长度和对应的累积比例）。
#
# 创建了一个新的图形，并设置了其大小。
#
# 使用plt.bar()函数创建了柱状图。
#
# 我们添加了x轴标签（"Length"），y轴标签（"Cumulative Proportion"）和图表标题。
#
# 在每个柱子顶部添加了比例值标签，保留两位小数。
#
# 调整了x轴的刻度，使其显示所有的长度值，并将标签旋转45度以避免重叠。
#
# 设置了y轴显示百分比格式。
#
# 最后，调用plt.tight_layout()来自动调整布局，并使用plt.show()显示图表。
#
# 这段代码将生成一个新的柱状图，展示每个长度及其以上的累积比例。横坐标是长度，纵坐标是累积比例（以百分比形式显示）。
# 	:param ast_deep_dict:
# 	:return:
# 	'''
# 	# Data
# 	# data = {13: 481, 21: 56, 8: 280, 11: 538, 7: 128, 15: 312, 10: 432, 20: 75, 14: 377, 9: 379, 19: 118, 16: 238,
# 	12: 478, 17: 184, 18: 154, 30: 4, 23: 42, 25: 14, 22: 37, 24: 20, 6: 6, 33: 2, 27: 14, 29: 5, 26: 8, 32: 1, 31: 3,
# 	28: 2, 42: 1, 39: 2, 34: 2, 5: 1}
# 	data = ast_deep_dict
# 	# Sort data by key (length)
# 	sorted_data = dict(sorted(data.items()))
#
# 	# Calculate total count
# 	total_count = sum(sorted_data.values())
#
# 	# Calculate cumulative proportions
# 	cumulative_proportions = {}
# 	for length in sorted_data.keys():
# 		cumulative_count = sum(value for key, value in sorted_data.items() if key >= length)
# 		cumulative_proportions[length] = cumulative_count / total_count
#
# 	# Prepare data for plotting
# 	lengths = list(cumulative_proportions.keys())
# 	proportions = list(cumulative_proportions.values())
#
# 	# Create bar chart
# 	plt.figure(figsize=(15, 8))
# 	bars = plt.bar(lengths, proportions)
#
# 	# Add labels and title
# 	plt.xlabel('Length')
# 	plt.ylabel('Cumulative Proportion')
# 	plt.title('Cumulative Proportion of Lengths')
#
# 	# Add value labels on top of each bar
# 	for bar in bars:
# 		height = bar.get_height()
# 		plt.text(bar.get_x() + bar.get_width() / 2., height,
# 		         f'{height:.2f}',
# 		         ha='center', va='bottom')
#
# 	# Adjust x-axis ticks
# 	plt.xticks(lengths, rotation=45)
#
# 	# Set y-axis to display percentages
# 	plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
#
# 	# Display the chart
# 	plt.tight_layout()
# 	plt.show()


def count_function_params ( source_code ) :
    ast = c_code_to_ast ( source_code )

    # print(ast)

    def extract_parameters ( json_data ) :
        # 解析JSON字符串
        data = json.loads ( json_data )

        # 寻找函数定义
        function_def = next (
                (child for child in data [ 'children' ] if child [ 'type' ] == 'function_definition') , None
        )

        if not function_def :
            return [ ]

        # 寻找函数声明器
        function_declarator = next (
                (child for child in function_def [ 'children' ] if child [ 'type' ] == 'function_declarator') , None
        )

        if not function_declarator :
            return [ ]

        # 寻找参数列表
        parameter_list = next (
                (child for child in function_declarator [ 'children' ] if child [ 'type' ] == 'parameter_list') ,
                None
        )

        if not parameter_list :
            return [ ]

        # 提取参数
        parameters = [ ]
        for child in parameter_list [ 'children' ] :
            if child [ 'type' ] == 'parameter_declaration' :
                param_type = next (
                        (c [ 'text' ] for c in child [ 'children' ] if c [ 'type' ] == 'primitive_type') , ''
                )
                param_name = next ( (c [ 'text' ] for c in child [ 'children' ] if c [ 'type' ] == 'identifier') , '' )
                parameters.append ( (param_type , param_name) )

        return parameters

    # print(extract_parameters(ast), len(extract_parameters(ast)))
    ast_deep_dict [ len ( extract_parameters ( ast ) ) ] = ast_deep_dict [ len ( extract_parameters ( ast ) ) ] + 1
    return len ( extract_parameters ( ast ) )


# # print(ast)
#
# # 找到 parameter_list 节点
# def find_parameter_list(data):
# 	if isinstance(data, dict):
# 		if data.get('type') == 'parameter_list':
# 			return data
# 		for value in data.values():
# 			result = find_parameter_list(value)
# 			if result:
# 				return result
# 	elif isinstance(data, list):
# 		for item in data:
# 			result = find_parameter_list(item)
# 			if result:
# 				return result
# 	return None
#
# parameter_list = find_parameter_list(ast)
#
# if not parameter_list:
# 	return [], 0
#
# # 计算参数个数和提取参数名
# params = []
# for child in parameter_list.get('children', []):
# 	if isinstance(child, dict) and child.get('type') == 'parameter_declaration':
# 		for subchild in child.get('children', []):
# 			if isinstance(subchild, dict) and subchild.get('type') == 'identifier':
# 				params.append(subchild.get('text', ''))
# 				break

# return params, len(params)


# 增加字段大小限制
csv.field_size_limit ( 2147483647 )  # 设置为2GB，你可以根据需要调整这个值
if __name__ == '__main__' :
    # 	# 示例代码
    # 	source_code = """
    # 	int add(int a, int b) {
    #     return a + b;
    # }
    # 	"""
    # 	count_function_params(source_code)  # 输出应为2
    # 	exit()

    input_file = 'RawDatasetsfunction_devign_less_len_400.csv'

    # 打开CSV文件
    # 以下这段代码的作用只是做了统计
    with open ( input_file , 'r' , encoding = 'utf-8' ) as file :
        # 创建CSV读取器
        csv_reader = csv.DictReader ( file )
        # 循环遍历每一行
        for row in csv_reader :
            # 获取"Complete Vulnerable Code"字段的内容
            vulnerable_code = row [ 'func' ]
            # 打印或处理vulnerable_code
            # print(vulnerable_code)
            # print('-' * 50)  # 分隔线，方便查看每段代码
            # count_function_params ( vulnerable_code )
            parse_code ( vulnerable_code )
        # 总和是2852
        print ( "键值对中值的和，即代码的总数是： " , sum ( ast_deep_dict.values ( ) ) )
        print ( "长度为键，函数的参数数量为值，该键值对为： " )
        print ( ast_deep_dict )
    visulizeASTStatistics ( ast_deep_dict )

    # 这是统计结果
    # defaultdict ( <class 'int'> , { 8 : 578 , 7: 305, 11: 396 , 10: 522 , 9: 608 , 14: 48 , 13: 133 , 12: 172 ,
    # 15: 31 , 6: 29 , 16: 18 , 20: 1 , 17: 7 , 18: 2 , 19: 1 , 39: 1 })
    transform_datasets_Triggers_ASTdepth ( input_file = input_file )
