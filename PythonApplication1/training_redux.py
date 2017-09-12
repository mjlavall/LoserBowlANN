from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter

ds = SupervisedDataSet(4, 1)
ds.addSample((-26,-9,-278,2),(1))
ds.addSample((148,18,346,1),(6))
ds.addSample((54,-47,166,1),(-4))
ds.addSample((115,13,1120,-1),(19))
ds.addSample((86,56,758,0),(9))
ds.addSample((25,-72,-175,1),(6))
ds.addSample((3,-38,286,-1),(-7))
ds.addSample((15,-1,50,3),(-9))
ds.addSample((21,-79,-50,1),(-1))
ds.addSample((-41,95,586,0),(-1))
ds.addSample((138,48,544,2),(2))
ds.addSample((12,-12,806,-1),(-1))
ds.addSample((2,-34,-72,0),(-4))
ds.addSample((-19,-14,-478,-2),(-2))
ds.addSample((-53,-92,-396,1),(-22))
ds.addSample((135,85,260,-1),(28))


net = buildNetwork(4, 5, 1, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)
result = trainer.trainUntilConvergence(maxEpochs=10000)
print(result)
NetworkWriter.writeToFile(net, 'LoserBowlANN_redux.xml')

test=net.activate((-126,35,-33,0.65))
print(test)