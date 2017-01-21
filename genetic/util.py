import random

def PrepareDataset(path, dtype):
    fd = open(path, 'r')
    rawdata = fd.readlines()
    fd.close()
    dataset = []
    for i in rawdata:
        tmp = i.strip("\n")
        dataset.append(dtype(tmp.strip("\r")))
    return (dataset)

def PrepareDatasetByDelimiter(path, dtype, delim):
    fd = open(path, 'r')
    rawdata = fd.readlines()
    fd.close()
    dataset1 = []
    dataset2 = []
    for i in rawdata:
        tmp = eval('['+i[:-1]+']')
        dataset1.append(dtype(tmp[0]))
        dataset2.append(dtype(tmp[1]))
    return (dataset1, dataset2)
    

def InitPopulation(dataset, length):
    gene = []
    for i in range(length):
        gene.append(random.choice(dataset))
    return (gene)

def PrintIndOfPopulation(ppl):
    for i in ppl.population:
    	i.Print()

def PrintIndOfList(ind):
    for i in ind:
    	i.Print()

def GetBestIndOfPopulation(ppl):
    maxfit  = ppl.population[0].fit
    minbox  = len(set(ppl.population[0].ind))
    bestind = ppl.population[0].ind
    for i in ppl.population:
        if (maxfit >= i.fit):
            if (minbox > len(set(i.ind))):
                minbox  = len(set(i.ind))
                bestind = i.ind
    return (bestind)

def GetBestFitnessOfPopulation(ppl):
    return ppl.population[0].fit

def GetWorstFitnessOfPopulation(ppl):
    return ppl.population[-1].fit

def GetAverageFitnessOfPopulation(ppl):
    totalfit = 0
    for i in ppl.population:
        totalfit += i.fit
    return totalfit/len(ppl.population)

def GetAllFitnessAsList(ppl):
    fitlist = []
    for i in ppl.population:
        fitlist.append(i.fit)
    return fitlist
