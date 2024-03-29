from .thrdnets import BiGNN, BiGNN_Submanifold, BiGNN_reproduce, BiGNN_reproduce_v1, BiGNN_Light, BiGNN_Light2, DBiGNN, SECONDNET
from .twodnets import PCDetBEVNet
threed_models = {'BiGNN': BiGNN, 'BiGNN_Submanifold':BiGNN_Submanifold, 
                'BiGNN_reproduce': BiGNN_reproduce, 'BiGNN_reproduce_v1':BiGNN_reproduce_v1, 'DBiGNN':DBiGNN,
                'BiGNN_Light': BiGNN_Light, 'BiGNN_Light2':BiGNN_Light2, 'SECONDNET':SECONDNET}
twod_models = {'PCDetBEVNet': PCDetBEVNet}