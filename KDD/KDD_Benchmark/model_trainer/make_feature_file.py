#encoding: utf-8
import pyprind
import util
from example import Example

'''
Make_feature_file函数:生成特征文件
输入参数解释:
authorIdPaperIds:
{
    参数类型: 列表,列表中的每个元素类型为AuthorIdPaperId类 (defined in authorIdPaperId.py)
    该类包含(作者id,论文id,label)三个类成员
}

dict_coauthor
{
    参数类型: 字典 generated from coauthor.json
    key类型为作者id
    val类型仍为字典,其格式为{a1:count1,a2:count2...ak:countk}
}

dict_paperIdAuthorId_to_name_aff
{
    参数类型: 字典 generated from dictionary from paperIdAuthorId_to_name_and_affiliation.json
    key类型为string,format为"PaperId|AuthorId"的复合形式
    value类型为字典,其格式为{"affiliation":"a1|a2|...|an","name":"n1|n2|...|nm"}
}

PaperAuthor
{
    参数类型: DataFrame, read from PaperAuthor.csv by pandas
    DateFrame包括的列有:
    PaperId     int     论文编号
    AuthorId    int     论文编号
    Name        string  作者名称
    Affiliation string  隶属单位
}

Author
{
    参数类型: DataFrame, read from Author.csv by pandas
    DateFrame包括的列有:
    Id          int     作者编号
    Name        string  作者名称
    Affiliation string  隶属单位
} 

feature_function_list: 列表,包含了所有特征函数的函数名称,形如[coauthor_1,coauthor_2,...]
to_file:生成文件路径和文件名
'''
def Make_feature_file(authorIdPaperIds, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference, feature_function_list, to_file):
    example_list = []
    dimension = 0

    process_bar = pyprind.ProgPercent(len(authorIdPaperIds))
    # 枚举列表中每一个AuthorIdPaperId类实例,转换成一个example实例
    for authorIdPaperId in authorIdPaperIds:
        process_bar.update()
        # 调用所有的特征函数生成若干个Feature类存放在features中
        # feature = [Feature1,Feature2,Feature3...]
        features = [feature_function(authorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Journal, Conference) for feature_function in feature_function_list]
        
        # 合并特征,关于mergeFeatures的解释可以参见util.py中的注释
        # 得到的feature的name为空,dimension为features中的dimension之和
        # feat_string为features中feat_string重排key值之后拼接得到
        feature = util.mergeFeatures(features)
        dimension = feature.dimension

        #特征target
        target = authorIdPaperId.label
        if target is None:
            target = "-1" #-1表示尚未分类
        
        #然后构造一个example类
        example = Example(target, feature)
        # example.comment = json.dumps({"paperId": authorIdPaperId.paperId, "authorId": authorIdPaperId.authorId})
        # comment则用paperId和authorId拼接得到
        example.comment = "%s %s" % (authorIdPaperId.paperId, authorIdPaperId.authorId)

        example_list.append(example)


    # 调用util.py中的write_example_list_to_file函数将example类写入特征文件
    # 函数的注释可在util.py中查看
    # 每一行的格式为 tar key1:val1 key2:val2 ... key_n:val_n # paperId authorId
    util.write_example_list_to_file(example_list, to_file)

    # 调用util.py中的write_example_list_to_arff_file函数将example类写入arff文件
    # 函数的注释可在util.py中查看
    # 此外关于arff文件的介绍也可以在util.py中定义该函数的上方找到
    util.write_example_list_to_arff_file(example_list, dimension, to_file+".arff")
