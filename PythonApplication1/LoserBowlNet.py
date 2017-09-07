from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(10, 2)
ds.addSample((-26,-9,-139,2,-1,26,9,139,-2,1),(1,-1))
ds.addSample((148,18,173,1,-1,-148,-18,-173,-1,1),(1,-1))
ds.addSample((54,-47,83,1,-1,-54,47,-83,-1,1),(-1,1))
ds.addSample((115,13,560,-1,-1,-115,-13,-560,1,1),(1,-1))
ds.addSample((86,56,379,0,-1,-86,-56,-379,0,1),(1,-1))
ds.addSample((25,-72,-175,1,-1,-25,72,175,-1,1),(1,-1))
ds.addSample((3,-38,143,-1,-1,-3,38,-143,1,1),(-1,1))
ds.addSample((15,-1,25,3,-1,-15,1,-25,-3,1),(-1,1))
ds.addSample((21,-79,-25,1,-1,-21,79,25,-1,1),(-1,1))
ds.addSample((-41,95,293,0,-1,41,-95,-293,0,1),(-1,1))
ds.addSample((138,48,272,2,-1,-138,-48,-272,-2,1),(1,-1))
ds.addSample((12,-12,806,-1,-1,-12,12,-806,1,1),(-1,1))
ds.addSample((2,-34,-36,0,-1,-2,34,36,0,1),(-1,1))
ds.addSample((-19,-14,-239,-2,-1,19,14,239,2,1),(-1,1))
ds.addSample((-53,-92,-198,1,-1,53,92,198,-1,1),(-1,1))
ds.addSample((135,85,130,-1,-1,-135,-85,-130,1,1),(1,-1))

net = buildNetwork(10, 7, 2, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)
result = trainer.trainUntilConvergence(maxEpochs=10000)
print(result)

test = [147,51,41,-1,-1,-147,-51,-41,1,1]
test2 = [-129,-127,-422,3,-1,129,127,422,-3,1]
predict = [44.6,16,11,11,-1,9.7,-13.5,-114,-20,1]
print(net.activate(test))
print([1,-1])
print(net.activate(test2))
print([-1,1])
print(net.activate(predict))
print([1,-1])

