# Algorithms

All algorithm configs here are an MVP and not tuned. All of them do not obtain good reward.

`bc.yaml`: currently bugged because rllib

`rainbow.yaml`: sometimes discovers rewarding episodes but rewards are too sparse to learn

`sac-discrete.yaml`: extremely slow to train

`sac-offline.yaml`: Q function drastically overestimates values due to OOD actions
