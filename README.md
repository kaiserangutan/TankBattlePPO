# TankBattlePPO

A custom competitive MARL environment TankBattle, along with PPO agent training/testing logic, and select trained checkpoints.

The TankBattle env is a 2D semi-physics based competitive game environment, where agents control the tanks' motion and firing. Projectiles not only have to collide with enemy tank, but "penetrate" the armor on that side to deal any damage. Armor on different sides of the tanks are different: front armor is hardest to penetrate, sides are easier, while the rear is most vulnerable.

Trained checkpoints exhibit behaviors such as armor angling and flanking to both protect their own tank against penetration, but attack the opponent's tank in the most effective way possible.
