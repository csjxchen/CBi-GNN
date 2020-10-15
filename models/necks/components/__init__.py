from .thrdnets import BiGNN, BiGNN_Submanifold, BiGNN_reproduce
from .twodnets import PCDetBEVNet
threed_models = {'BiGNN': BiGNN, 'BiGNN_Submanifold':BiGNN_Submanifold, 'BiGNN_reproduce': BiGNN_reproduce}
twod_models = {'PCDetBEVNet': PCDetBEVNet}