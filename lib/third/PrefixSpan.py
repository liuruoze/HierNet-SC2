'''
PrefixSpan算法实现

Author: 西木野Maki/holy5pb@163.com
'''

data = [
    [1,4,2,3],
    [0, 1, 2, 3],
    [1, 2, 1, 3, 4],
    [2, 1, 2, 4, 4],
    [1, 1, 1, 2, 3],
]

def rmLowSup(D,Cx, postfixDic,minSup):
    '''
    根据当前候选集合删除低support的候选集
    '''
    Lx = Cx
    for iset in Cx:
        if len(postfixDic[iset])/len(D) < minSup:
            Lx.remove(iset)
            postfixDic.pop(iset)
        
    return Lx, postfixDic

def createC1(D, minSup):
    '''生成第一个候选序列，即长度为1的集合序列

    '''
    C1 = []
    postfixDic={}  # 集合对应的每个序列的所在位置的开始位置，{(1,):{0:2,3:1}}表示前缀(1,)在0序列的结尾为2， 3序列的结尾为1
    lenD = len(D)

    for i in range(lenD):
        for idx, item in enumerate(D[i]):
            if tuple([item]) not in C1:
                postfixDic[tuple([item])]={}
                
                C1.append(tuple([item]))
            if i not in postfixDic[tuple([item])].keys():
                postfixDic[tuple([item])][i]=idx

    L1, postfixDic = rmLowSup(D, C1, postfixDic, minSup)              
    
    return L1, postfixDic

def genNewPostfixDic(D,Lk, prePost):
    '''根据候选集生成相应的PrefixSpan后缀

    参数：
        D：数据集
        Lk: 选出的集合
        prePost: 上一个前缀集合的后缀

    基于假设：
        (1,2)的后缀只能出现在 (1,)的后缀列表里 
    e.g.
        postifixDic[(1,2)]={1:2,2:3}  前缀 (1,2) 在第一行的结尾是2，第二行的结尾是3
    '''
    postfixDic = {}

    for Ck in Lk:
        # (1,2)的后缀只能出现在 (1,)的后缀列表里
        postfixDic[Ck]={}
        tgt = Ck[-1]
        prePostList = prePost[Ck[:-1]]
        for r_i in prePostList.keys():
            for c_i in range(prePostList[r_i]+1, len(D[r_i])):
                if D[r_i][c_i]==tgt:
                    postfixDic[Ck][r_i] = c_i
                    break
    return postfixDic

def psGen(D, Lk, postfixDic, minSup, minConf):
    '''生成长度+1的新的候选集合
    '''

    retList = []
    lenD = len(D)
    # 访问每一个前缀
    for Ck in Lk:
        item_count = {}  # 统计item在多少行的后缀里出现
        # 访问出现前缀的每一行
        for i in postfixDic[Ck].keys():
            # 从前缀开始访问每个字符
            item_exsit={}
            for j in range(postfixDic[Ck][i]+1, len(D[i])):
                if D[i][j] not in item_count.keys():
                    item_count[D[i][j]]=0
                if D[i][j] not in item_exsit:
                    item_count[D[i][j]]+=1
                    item_exsit[D[i][j]]=True

        c_items = []

        # 根据minSup和minConf筛选候选字符
        for item in item_count.keys():
            if item_count[item]/lenD >= minSup and item_count[item]/len(postfixDic[Ck])>=minConf:
                c_items.append(item)
        
        # 将筛选后的字符组成新的候选项，加入候选集合
        for c_item in c_items:
            retList.append(Ck+tuple([c_item]))
    
    return retList

def PrefixSpan(D, minSup=0.5, minConf=0.5, sptNum=False):
    L1, postfixDic = createC1(D, minSup)

    Dic_L1 = [{x:len(postfixDic[x])} for x in L1]
    
    Dic_L = [Dic_L1]

    L = [L1]

    k=2
    while len(L[k-2])>0:
        # 生成新的候选集合
        Lk = psGen(D, L[k-2], postfixDic,minSup, minConf)
        # 根据新的候选集合获取新的后缀
        postfixDic = genNewPostfixDic(D,Lk,postfixDic)
        Dic_Lk = [{x:len(postfixDic[x])} for x in Lk]

        # 加入到集合中
        L.append(Lk)
        Dic_L1.append(Dic_Lk)
        k+=1
    
    return Dic_L


if  __name__ =='__main__':

    L = PrefixSpan(data,minSup=0.8,minConf=0, sptNum=True)
    for i in range(len(L)-1,-1,-1):
        for Cx in L[i]:
            print(Cx)