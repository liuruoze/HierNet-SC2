# PPO param
# gamma = 1
# lamda = 1
# clip_value = 0.1
# c_1 = 0.01   # value network
# c_2 = 1e-5  # entropy

gamma = 0.9995   # 0.9995 the same important as lamda
lamda = 0.9995   # important, change to 0.99 will make training failed
clip_value = 0.2
c_1 = 0.5  # 0.01
c_2 = 1e-3  # entropy 1e-3

# batch_size
batch_size = 512

# base_learning_rate
lr = 1e-4

# network
update_num = [20, 20, 20, 20, 20]

# whether to use true return_values for value network predict
use_return_error = True

# whether to use hierarchical learing rate, "the controller needs to wait the below module updates first"
use_hier_lr = False
if use_hier_lr:
    lr_wieght = [1, 0.5, 0.1, 0.1, 0.5]
else:
    lr_wieght = [1, 1, 1, 1, 1]
lr_list = [i * lr for i in lr_wieght]

# whether to use hierarchical reward, "the higher level, the more reward weight for the module"
# also know as a "The greater the power, the greater the responsibility"
use_hier_reward = False
if use_hier_reward:
    reward_weight = [1, 0.7, 0.5, 0.5, 0.7]
else:
    reward_weight = [1, 1, 1, 1, 1]

# whether to seperate policy and value
use_dual_policy_value = True

# whether to use alternative update
use_alternative_update = False

# below value will be overridedd in main.py
restore_model = False
