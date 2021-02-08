REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_influence import BasicMACInfluence
from .basic_controller_interactive import BasicMACInteractive, BasicMACInteractiveRegV1, BasicMACInteractiveRegV2
from .rode_controller import RODEMAC
from .separate_controller import SeparateMAC
from .entity_controller import EntityMAC 
from .basic_controller_policy import BasicMAC as PolicyMAC
from .central_basic_controller import CentralBasicMAC  
from .noise_controller import NoiseMAC

from .cate_broadcast_comm_controller import CateBCommMAC
from .cate_broadcast_comm_controller_full import CateBCommFMAC
from .cate_broadcast_comm_controller_not_IB import CateBCommNIBMAC
from .tar_comm_controller import TarCommMAC
from .cate_pruned_broadcast_comm_controller import CatePBCommMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_mac_interactive"] = BasicMACInteractive                      #   for project CollaQ
REGISTRY["basic_mac_interactive_regv1"] = BasicMACInteractiveRegV1           #   for project CollaQ
REGISTRY["basic_mac_interactive_regv2"] = BasicMACInteractiveRegV2           #   for project CollaQ
REGISTRY["basic_mac_influence"] = BasicMACInfluence                         
REGISTRY['rode_mac'] = RODEMAC        # for project RODE
REGISTRY["separate_mac"] = SeparateMAC     # for project ROMA
REGISTRY["entity_mac"] = EntityMAC     # for project AIQMIX
REGISTRY["policy"] = PolicyMAC                               # for project WQMIX
REGISTRY["basic_central_mac"] = CentralBasicMAC              # for project WQMIX
REGISTRY["noise_mac"] = NoiseMAC        # for project MAVEN



REGISTRY = {
        "basic_mac":BasicMAC,
        "basic_mac_interactive":BasicMACInteractive,                      #   for project CollaQ
        "basic_mac_interactive_regv1":BasicMACInteractiveRegV1,           #   for project CollaQ
        "basic_mac_interactive_regv2":BasicMACInteractiveRegV2,           #   for project CollaQ
        "basic_mac_influence":BasicMACInfluence,                         
        'rode_mac':RODEMAC,                                   # for project RODE
        "separate_mac":SeparateMAC,                         # for project ROMA
        "entity_mac":EntityMAC,                             # for project AIQMIX
        "policy":PolicyMAC,                                # for project WQMIX
        "basic_central_mac":CentralBasicMAC,               # for project WQMIX
        "noise_mac":NoiseMAC,                               # for project MAVEN
        "cate_broadcast_comm_mac": CateBCommMAC,            # for project NDQ
        "cate_broadcast_comm_mac_full": CateBCommFMAC,      # for project NDQ
        "cate_broadcast_comm_mac_not_IB": CateBCommNIBMAC,  # for project NDQ
        "tar_comm_mac": TarCommMAC,                         # for project NDQ
        "cate_pruned_broadcast_comm_mac": CatePBCommMAC,    # for project NDQ
}    
