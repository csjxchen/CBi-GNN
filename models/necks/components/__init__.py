from .thrdnets import BiGNN, BiGNN_Submanifold
from .twodnets import PCDetBEVNet
threed_models = {'BiGNN': BiGNN, 'BiGNN_Submanifold':BiGNN_Submanifold}
twod_models = {'PCDetBEVNet': PCDetBEVNet}