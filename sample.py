import utils
def main() :
    datasets = [['IceCream','JellyBean','Kitkat','Lollipop'],
                ['EQ','JDT','LC','ML','PDE'],
                ['apache','safe','zxing'],
                ['ant-1.3','arc','camel-1.0','poi-1.5','redaktor','skarbonka','tomcat','velocity-1.4','xalan-2.4','xerces-1.2'],
                ['CM1','MW1','PC1','PC3','PC4'],
                ['JM1'],
                ['PC2'],
                ['PC5','MC1'],
                ['MC2','KC3'],
                ['ar1','ar3','ar4','ar5','ar6']]

    all_auc = {}
    for i in range(10) :
        for j in range(len(datasets[i])) :
            file2 = datasets[i][j]
            all_auc[file2] = {}
            for k in range(10) :
                if k == i :
                    continue
                for l in range(len(datasets[k])) :
                    file1 = datasets[k][l]
                    get_auc = utils.main_step(file1,file2)
                    if get_auc != -1:
                        all_auc[file2][file1] = get_auc
    utils.generateResults(all_auc)
    return all_auc


all_auc = main()
    
