from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

_LOAD_MODEL_PATH = "./model/20180621-134211/"
_SAVE_MODEL_PATH = "./model/"

# define the num of input and output
_SIZE_HIGH_NET_INPUT = 20
_SIZE_HIGH_NET_OUT = 3

_SIZE_CONTROLLER_OUT = 2
_SIZE_BASE_NET_OUT = 2

_SIZE_TECH_NET_INPUT = 9
_SIZE_TECH_NET_OUT = 4

_SIZE_POP_NET_INPUT = 12
_SIZE_POP_NET_OUT = 3

_SIZE_BATTLE_NET_OUT = 2

_SIZE_FIGHT_NET_OUT = 3


MAP_CHANNELS = 10

# timesteps per second
_FPS = 22.4

# Minimap index
_M_HEIGHT = features.MINIMAP_FEATURES.height_map.index
_M_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_M_CAMERA = features.MINIMAP_FEATURES.camera.index
_M_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_M_SELECTED = features.MINIMAP_FEATURES.selected.index


# Screen index
_S_HEIGHT = features.SCREEN_FEATURES.height_map.index
_S_VISIBILITY = features.SCREEN_FEATURES.visibility_map.index
_S_POWER = features.SCREEN_FEATURES.power.index
_S_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_S_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_S_SELECTED = features.SCREEN_FEATURES.selected.index
_S_HITPOINT_R = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_S_SHIELD_R = features.SCREEN_FEATURES.unit_shields_ratio.index
_S_DENSITY_A = features.SCREEN_FEATURES.unit_density_aa.index

# Unit type index
_MINERAL_TYPE_INDEX = 483
_GAS_TYPE_INDEX = 342

_PROBE_TYPE_INDEX = 84
_ZEALOT_TYPE_INDEX = 73
_STALKER_TYPE_INDEX = 74

_NEXUS_TYPE_INDEX = 59
_PYLON_TYPE_INDEX = 60
_ASSIMILATOR_TYPE_INDEX = 61
_FORGE_TYPE_INDEX = 63
_CANNON_TYPE_INDEX = 66
_GATEWAY_TYPE_INDEX = 62
_CYBER_TYPE_INDEX = 72

UNIT_MAP = {
    84: "Probe",
    73: "Zealot",
    74: "Stalker",
    59: "Nexus",
    62: "Gateway",
    60: "Pylon",
    61: "Assimilator",
    342: "VespeneGeyser",
    483: "MineralField750",
    341: "MineralField", 
    1961: "MineralField450",
    88: "Extractor",
    20: "Refinery",

}

UNIT_MAP_INV = {v: k for k, v in UNIT_MAP.items()}

# _M_RELATIVE_TYPE
_RELATIVE_NONE = 0
_RELATIVE_SELF = 1
_RELATIVE_ALLY = 2
_RELATIVE_NEUTRAL = 3
_RELATIVE_ENEMY = 4

# Action type index
_NO_OP = sc2_actions.FUNCTIONS.no_op.id
_SMART_SCREEN = sc2_actions.FUNCTIONS.Smart_screen.id
_SELECT_ARMY = sc2_actions.FUNCTIONS.select_army.id
_SELECT_WORKER = sc2_actions.FUNCTIONS.select_idle_worker.id
_SELECT_BY_ID = sc2_actions.FUNCTIONS.select_unit.id
_CONTROL_GROUP = sc2_actions.FUNCTIONS.select_control_group.id

_ATTACH_M = sc2_actions.FUNCTIONS.Attack_minimap.id
_ATTACK_S = sc2_actions.FUNCTIONS.Attack_screen.id
_MOVE_S = sc2_actions.FUNCTIONS.Move_screen.id
_MOVE_M = sc2_actions.FUNCTIONS.Move_minimap.id

_SELECT_UNIT = sc2_actions.FUNCTIONS.select_unit.id
_SELECT_POINT = sc2_actions.FUNCTIONS.select_point.id
_MOVE_CAMERA = sc2_actions.FUNCTIONS.move_camera.id

_TRAIN_PROBE = sc2_actions.FUNCTIONS.Train_Probe_quick.id
_TRAIN_ZEALOT = sc2_actions.FUNCTIONS.Train_Zealot_quick.id
_TRAIN_STALKER = sc2_actions.FUNCTIONS.Train_Stalker_quick.id

_BUILD_PYLON_S = sc2_actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_ASSIMILATOR_S = sc2_actions.FUNCTIONS.Build_Assimilator_screen.id
_BUILD_FORGE_S = sc2_actions.FUNCTIONS.Build_Forge_screen.id
_BUILD_GATEWAY_S = sc2_actions.FUNCTIONS.Build_Gateway_screen.id
_BUILD_CYBER_S = sc2_actions.FUNCTIONS.Build_CyberneticsCore_screen.id

_HARVEST_S = sc2_actions.FUNCTIONS.Harvest_Gather_screen.id

_A_SMART_SCREEN = sc2_actions.FUNCTIONS.Smart_screen.ability_id
_A_TRAIN_PROBE = sc2_actions.FUNCTIONS.Train_Probe_quick.ability_id
_A_TRAIN_ZEALOT = sc2_actions.FUNCTIONS.Train_Zealot_quick.ability_id
_A_TRAIN_STALKER = sc2_actions.FUNCTIONS.Train_Stalker_quick.ability_id

_A_BUILD_PYLON_S = sc2_actions.FUNCTIONS.Build_Pylon_screen.ability_id
_A_BUILD_ASSIMILATOR_S = sc2_actions.FUNCTIONS.Build_Assimilator_screen.ability_id
_A_BUILD_FORGE_S = sc2_actions.FUNCTIONS.Build_Forge_screen.ability_id
_A_BUILD_GATEWAY_S = sc2_actions.FUNCTIONS.Build_Gateway_screen.ability_id
_A_BUILD_CYBER_S = sc2_actions.FUNCTIONS.Build_CyberneticsCore_screen.ability_id

_A_ATTACK_ATTACK_MINIMAP_S = sc2_actions.FUNCTIONS.Attack_Attack_minimap.ability_id
_A_ATTACK_MINIMAP_S = sc2_actions.FUNCTIONS.Attack_minimap.ability_id
_A_ATTACK_ATTACK_SCREEN_S = sc2_actions.FUNCTIONS.Attack_Attack_screen.ability_id
_A_ATTACK_SCREEN_S = sc2_actions.FUNCTIONS.Attack_screen.ability_id


_NOT_QUEUED = [0]
_QUEUED = [1]

_CLICK = [0]
_SHIFT_CLICK = [1]
_DBL_CLICK = [2]

_RECALL_GROUP = [0]
_SET_GROUP = [1]
_APPEND_GROUP = [2]

_GATEWAY_GROUP_ID = [9]
_BASE_GROUP_ID = [0]
_ARMY_GROUP_ID = [3]

_ARMY_INDEX = -1
_GATEWAY_GROUP_INDEX = -9

# up, up_right, right, right_down, down, down_left, left, left_up
center_x = 32
center_y = 32
movement = 15
move_pos_array = [[center_x, center_y - movement], [center_x + movement, center_y - movement],
                  [center_x + movement, center_y], [center_x + movement, center_y + movement],
                  [center_x, center_y + movement], [center_x - movement, center_y + movement],
                  [center_x - movement, center_y], [center_x - movement, center_y - movement],
                  ]

# screen pos
# mineral_pos = [18, 26]
# gas1_pos = [18, 38]
# gas2_pos = [45, 11]
# base_pos = [36, 35]

# minimap pos
my_sub_pos = [41, 20]      # our sub mineral pos
#enemy_sub_pos = [13, 50]
#enemy_main_pos = [41, 45]

enemy_sub_pos = [13, 50]
enemy_main_pos = [45, 47]

base_camera_pos = [19, 24]


# game difficulty
difficulty = 1


def time_wait(sec):
    return int(sec * _FPS)
