from .thrdnets import BiGNN, BiGNN_Submanifold, BiGNN_reproduce, BiGNN_reproduce_v1
from .twodnets import PCDetBEVNet
threed_models = {'BiGNN': BiGNN, 'BiGNN_Submanifold':BiGNN_Submanifold, 'BiGNN_reproduce': BiGNN_reproduce, 'BiGNN_reproduce_v1':BiGNN_reproduce_v1}
twod_models = {'PCDetBEVNet': PCDetBEVNet}