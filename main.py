#coding: utf-8

"""
GA(Genetic Algorithm, 遺伝的アルゴリズム)
Main Class

Create: 2016/11/27
Update: 2016/12/31
"""
import argparse
import random
import matplotlib.pyplot  as     plt
from   genetic.individual import Individual
from   genetic.population import Population
from   genetic.select     import Select
from   genetic.crossover  import Crossover
from   genetic.mutation   import Mutation
from   genetic.util       import *

def calcFit(args):
    """
    Create Calculation Fitness Function
    """
    ind             = args[0]
    weight          = args[1]
    AllowableAmount = args[2]
    fits = []
    for i in set(ind):
        indexes = [j for j, x in enumerate(ind) if x==i]
        gweight = sum([weight[j] for j in indexes])
        # {!}許容量超えの時にその個体の適応度を調整する必要がある
        loss    = AllowableAmount-gweight
        if (loss<0):
            loss += sum(weight)
        fits.append(abs(loss))

    return sum(fits)

def main():
    #引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--revo  '   ,  dest='revo'         , type=int,   default=40,   help='revolution count>0')
    parser.add_argument('--popcnt  ' ,  dest='popcnt'       , type=int,   default=300,  help='population count>0')
    parser.add_argument('--maxormin  ', dest='maxormin'     , type=str,   default="min",help='max or min')
    parser.add_argument('--esize  '  ,  dest='eliteSize'    , type=int,   default=1,    help='elite size>=0')
    parser.add_argument('--tsize  '  ,  dest='tornSize'     , type=int,   default=2,    help='tornament size>=2')
    parser.add_argument('--ctype  '  ,  dest='cross_type'   , type=str,   default="two",help='crossover type one or two or random')
    parser.add_argument('--cprob  '  ,  dest='cross_prob'   , type=float, default=0.6,  help='crossover probability>=0.0')
    parser.add_argument('--mprob  '  ,  dest='mutation_prob', type=float, default=0.05, help='mutation probability>=0.0')
    parser.add_argument('--dataset  ',  dest='dataset'      , type=str,   default="",   help='dataset file path')
    parser.add_argument('--dtype  '  ,  dest='dtype'        , type=str,   default="int",help='dataset type integer or float')
    parser.add_argument('--graph  '  ,  dest='graph'        , type=int,   default=0,    help='show graph > 0')

    """
    setup argment
    """
    args = parser.parse_args()
    if (args.revo         <0   or 
        args.popcnt       <0   or 
        args.maxormin    ==""  or
        args.eliteSize    <0   or 
        args.tornSize     <2   or 
        args.cross_prob   <0.0 or 
        args.mutation_prob<0.0 or 
        args.dataset     == "" or
        args.dtype       == "" or
        args.graph        <0):
        print("argment parse Error!")
        exit()

    revo          = args.revo
    popcnt        = args.popcnt
    maxormin      = args.maxormin
    eliteSize     = args.eliteSize
    tornSize      = args.tornSize
    cross_prob    = args.cross_prob
    mutation_prob = args.mutation_prob
    
    """
    prepare dataset
    """
    dataset, weight = PrepareDatasetAsDelimiter(path=args.dataset, dtype=int, delim=',')
    del dataset[-1]
    AllowableAmount = weight[-1]
    del weight[-1]
    gene = len(dataset)

    """
    init population
    """
    life = []
    for i in range(popcnt):
        g = []
        map(g.append,InitPopulation(dataset, gene))
        life.append(g)

    population = []
    for i in range(popcnt):
        ind = Individual(i+1)
        ind.CreateIndividual(life[i], 0)
        fit = ind.CalcFitness(calcFit, (ind.ind, weight, AllowableAmount))
        ind.SetFitness(fit)
        population.append(ind)

    ppl = Population()
    ppl.CreatePopulation(population)
    ppl.SortInFitness(maxormin)

    select    = Select(eliteSize, tornSize)
    cross     = Crossover(cross_prob)
    mut       = Mutation(dataset, mutation_prob)
    if   (args.cross_type == "two"):
        cross_type = cross.twoPoints
    elif (args.cross_type == "random"):
        cross_type = cross.randomPoints
    else:
        cross_type = cross.onePoint
    
    if (args.graph>0):
        count    = []
        bestfit  = []
        worstfit = []
        avefit   = []

    """
    revolution
    """
    for i in range(revo):
        print("revolution: {0}".format(i))
        #print("<DEBUG>ppl is")
        #PrintIndOfPopulation(ppl)
        # select
        """
        select elite from population
        """
        elite     = select.SelectElite(ppl)
        next_gene = elite
        #print("<DEBUG>elite is")
        #PrintIndOfList(next_gene)
        while (len(next_gene)<popcnt):
            """
            select offspring from population (tornament)
            """
            
            offspring = select.SelectTornament(ppl)
            print("<DEBUG>select is")
            #for j in range(len(offspring)):
                #fit = offspring[j].CalcFitness(calcFit, offspring[j].ind)
                #offspring[j].SetFitness(fit)
            
            print PrintIndOfList(offspring)
            """
            crossover from offspring
            """
            if (args.cross_type == "random"):
                cross_offspring    = cross_type(offspring)
                #print("<DEBUG>crossover is")
                #PrintIndOfList(cross_offspring)
            else:
                cross_offspring    = cross_type(offspring)
                #print("<DEBUG>crossover is")
                #PrintIndOfList(cross_offspring)

            """
            mutation from offspring
            """
            mutation_offspring = mut.Mutation(cross_offspring)
            print("<DEBUG>mutation is")
            PrintIndOfList(mutation_offspring)
            for j in range(len(mutation_offspring)):
                fit = mutation_offspring[j].CalcFitness(calcFit, (mutation_offspring[j].ind, weight, AllowableAmount))
                mutation_offspring[j].SetFitness(fit)
                next_gene.append(mutation_offspring[j])
            #print("<DEBUG>next_gene is")
            #PrintIndOfList(next_gene)
            
        """
        create new population
        """
        ppl.CreatePopulation(next_gene)
        ppl.SortInFitness(maxormin)
        #print("<DEBUG>ppl is")
        #PrintIndOfPopulation(ppl)

        if (args.graph>0):
            if ((i%1)==0):
                count.append(i)
                bestfit.append(GetBestFitnessOfPopulation(ppl))
                worstfit.append(GetWorstFitnessOfPopulation(ppl))
                avefit.append(GetAverageFitnessOfPopulation(ppl))
                #DEBUG
                left    = [i+1 for i in range(GetWorstFitnessOfPopulation(ppl), GetBestFitnessOfPopulation(ppl)+1)]
                height  = GetAllFitnessAsList(ppl)
                height  = [height.count(i) for i in left]
                plt.title("Population")
                plt.bar(left, height, align="center")
                plt.show()


    print "BestInd        :", GetBestIndOfPopulation(ppl)
    print "weight         :", weight
    print "AllowableAmount:", AllowableAmount
    print "BestFit        :", GetBestFitnessOfPopulation(ppl)
    print "WorstFit       :", GetWorstFitnessOfPopulation(ppl)
    print "AverageFit     :", GetAverageFitnessOfPopulation(ppl)
    if (args.graph>0):
        plt.title("Fittness")
        p1 = plt.plot(count, bestfit , linewidth=2)
        p2 = plt.plot(count, worstfit, linewidth=2, linestyle="dashed")
        p3 = plt.plot(count, avefit  , linewidth=2, linestyle="dashdot")
        plt.legend((p1[0], p2[0], p3[0]), ("Best", "Worst", "Average"), loc=2)
        plt.show()
        left    = [i+1 for i in range(GetWorstFitnessOfPopulation(ppl), GetBestFitnessOfPopulation(ppl)+1)]
        height  = GetAllFitnessAsList(ppl)
        height  = [height.count(i) for i in left]
        plt.title("Population")
        plt.bar(left, height, align="center")
        plt.show()
if __name__ == "__main__":
    main()
