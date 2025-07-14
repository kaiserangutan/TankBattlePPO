"""
TankBattleEnv, a 2D multi‑agent tank combat environment (v0.5)
Author: Haoran Qin, McGill University

"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import gym
import numpy as np
import pygame
from gym import spaces



MapW, MapH = 800, 600
Vec2 = pygame.math.Vector2
_DEG2RAD = math.pi / 180.0


@dataclass
class Projectile:
    pos: Vec2
    dir: Vec2  # unit vector
    speed: float = 35.0
    owner: int = 0  # 0 = agent, 1 = opponent
    def forward_vec(self) -> Vec2:
        return self.dir


@dataclass
class Tank:
    pos: Vec2
    speed: float
    hull_angle: float  # degrees, 0 = +x world axis (to the right)
    turret_angle: float  # global angle, not relative
    hp: int = 4
    cooldown: int = 0  # frames until next shot allowed

    # Movement constants
    FWD_ACCELERATION: float = 1.5
    FRICTION = 0.12 # lose speed portion / step
    MAX_SPEED = 9.0
    ROT_SPEED: float = 10.0  # deg / step
    TURRET_SPEED: float = 4.0  # deg / step
    SIZE: float = 25.0  # radius used for collisions & sprite size

    def forward_vec(self) -> Vec2:
        rad = self.hull_angle * _DEG2RAD
        return Vec2(math.cos(rad), math.sin(rad))

    def turret_vec(self) -> Vec2:
        rad = self.turret_angle * _DEG2RAD
        return Vec2(math.cos(rad), math.sin(rad))
    def collide_with_tank_at(self, point: Vec2) -> bool:
        return(self.pos - point).magnitude() < 2.8 * self.SIZE



class TankBattleEnv(gym.Env):
    """OpenAI‑Gym / Pygame tank duel, 17‑D obs, discrete action space."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}


    MOVE_FWD, MOVE_BACK, ROT_HULL_L, ROT_HULL_R, ROT_TUR_L, ROT_TUR_R, FIRE, NOOP = range(8)

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)


        self.walls = self._build_walls()


        self.agent: Tank
        self.enemy: Tank
        self.projectiles: List[Projectile]
        self.step_count: int
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self._flag = np.random.choice([True, False])


    def random_spawn_point(self):
        return [Vec2(np.random.randint(35, 180), np.random.randint(35, 180)), Vec2(np.random.randint(600, 765), np.random.randint(420, 565))]
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        tanks_colliding = True
        '''
        pygame.Rect(200, 480, 20, 120),
        pygame.Rect(600, 0, 20, 120),
        pygame.Rect(390, 310, 20, 120),
        pygame.Rect(290, 290, 220, 20),
        '''
        while tanks_colliding:
            _spawns = self.random_spawn_point()
            # if np.random.choice([True, False]):
            if self._flag:

                self.agent = Tank(pos=_spawns[0], speed=0, hull_angle = random.randint(0, 9), turret_angle=random.randint(0, 9))
                self.enemy = Tank(pos=_spawns[1], speed=0, hull_angle = random.randint(180, 189), turret_angle=random.randint(180, 189))
            else:
                self.agent = Tank(pos=_spawns[1], speed=0, hull_angle = random.randint(180, 189), turret_angle=random.randint(180, 189))
                self.enemy = Tank(pos=_spawns[0], speed=0, hull_angle = random.randint(0, 9), turret_angle=random.randint(0, 9))
            self._flag = not self._flag
            tanks_colliding = self.agent.collide_with_tank_at(self.enemy.pos)
        # print(self.agent.pos.x < 400)
        self.projectiles = []
        self.step_count = 0

        obs, obs_enemy = self._get_obs()
        if self.render_mode == "human":
            self._init_render()
            self.render()
        return obs, obs_enemy, {}
    def _reset_debug_mode(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        tanks_colliding = True
        while tanks_colliding:
            self.agent = Tank(pos=Vec2(100, 250), speed=0, hull_angle = 30, turret_angle=2)
            self.enemy = Tank(pos=Vec2(700, 210), speed=0, hull_angle = 90, turret_angle=180)
            tanks_colliding = self.agent.collide_with_tank_at(self.enemy.pos)
        self.projectiles = []
        self.step_count = 0

        obs, obs_enemy = self._get_obs()
        if self.render_mode == "human":
            self._init_render()
            self.render()
        return obs, obs_enemy, {}

    def step(self, action: int, enemy_action: int):
        self.step_count += 1
        reward = 0.0
        done = False


        reward += self._apply_action(self.agent, self.enemy, action)
        self._apply_action(self.enemy, self.agent, enemy_action)


        reward += self._update_projectiles()


        reward += self._line_of_fire_reward(self.agent, self.enemy)
        reward -= self._line_of_fire_reward(self.enemy, self.agent)


        if self.step_count >= 8000:
            done = True
        elif self.agent.hp <= 0:
            reward += -1300.0
            done = True
        elif self.enemy.hp <= 0:
            reward += 1300.0
            done = True
        obs, obs_enemy = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, obs_enemy, reward, done, {}



    def _apply_action(self, tank: Tank, tank2: Tank, action: int) -> float:
        if action == self.MOVE_FWD:
            tank.speed += tank.FWD_ACCELERATION if tank.speed < tank.MAX_SPEED else 0
        elif action == self.MOVE_BACK:
            tank.speed -= 1.1 * tank.FWD_ACCELERATION if tank.speed > 0 - tank.MAX_SPEED else 0
        elif action == self.ROT_HULL_L:
            tank.hull_angle = (tank.hull_angle - tank.ROT_SPEED) % 360
        elif action == self.ROT_HULL_R:
            tank.hull_angle = (tank.hull_angle + tank.ROT_SPEED) % 360
        elif action == self.ROT_TUR_L:
            tank.turret_angle = (tank.turret_angle - tank.TURRET_SPEED) % 360
        elif action == self.ROT_TUR_R:
            tank.turret_angle = (tank.turret_angle + tank.TURRET_SPEED + 1) % 360
        elif action == self.FIRE and tank.cooldown == 0:
            muzzle = tank.pos + tank.turret_vec() * (tank.SIZE - 2)
            self.projectiles.append(Projectile(muzzle, tank.turret_vec().rotate(np.random.normal(0, 1.2)), owner=tank))
            tank.cooldown = 20
        # reduce cooldown each frame
        self._try_move(tank, tank2, tank.forward_vec() * tank.speed)
        tank.speed *= 1 - tank.FRICTION
        if tank.cooldown > 0:
            tank.cooldown -= 1
        return -0.05 if 30 < self.agent.pos.x < 770 and 30 < self.agent.pos.y < 570 else -0.1

    def _try_move(self, tank: Tank, tank2: Tank, delta: Vec2):
        """Attempt to move delta."""
        new_pos = tank.pos + delta
        bbox = pygame.Rect(new_pos.x - tank.SIZE * 1.3, new_pos.y - tank.SIZE * 1.3, tank.SIZE * 2.6, tank.SIZE * 2.6)
        if (20 < new_pos[0] < 780) and (20 < new_pos[1] < 580) and (not tank2.collide_with_tank_at(new_pos)) and (not any(bbox.colliderect(w) for w in self.walls)):
            tank.pos = new_pos  # legal move
        else:
            self._try_move(tank, tank2, (delta * 0.5) if (delta * 0.5).magnitude() > 1.3 else Vec2(0,0))
            
        # If blocked, slow down recursively until acceptable


    def _update_projectiles(self) -> float:
        r = 0.0
        survivors = []
        for p in self.projectiles:
            
            p.pos += p.dir * p.speed
            hit_check_pts = [p.pos - 2 * i * p.forward_vec() for i in reversed(range(20))]
            # Out of bounds
            if not (0 < p.pos.x < MapW and 0 < p.pos.y < MapH):
                #r -= 0.005
                continue
            # Wall collision
            elif any(any(w.collidepoint(pt[0], pt[1]) for w in self.walls) for pt in hit_check_pts):
                #r -= 0.005
                continue

            # Collision with tanks
            target_tank = self.enemy if p.owner == self.agent else self.agent


            # Check hit against target
            if any([(p.owner != target_tank and (pt - target_tank.pos).length() < target_tank.SIZE * 1.3 and self._check_penetration(p, pt, target_tank)) for pt in hit_check_pts]):
                target_tank.hp -= 1
                r += 100.0 if target_tank == self.enemy else -100.0
                continue
            elif any(p.owner != target_tank and (pt - target_tank.pos).length() < target_tank.SIZE * 1.3 for pt in hit_check_pts):
            #     r += 10.0 if target_tank == self.agent else 0.0
                continue
            '''
            for pt in hit_check_pts:
                if p.owner != target_tank and (pt - target_tank.pos).length() < target_tank.SIZE * 1.3:
                    if self._check_penetration(p, pt, target_tank):
                        target_tank.hp -= 1
                        reward += 100.0 if target_tank == self.enemy else -100.0
                    else:
                        reward += 50.0 if target_tank == self.agent else 0.0
                continue
            '''

            survivors.append(p)
        self.projectiles = survivors
        return r



    def _check_penetration(self, p: Projectile, pt: Vec2, target: Tank) -> bool:
        incoming = ((pt - target.pos).angle_to(target.forward_vec()) + 360) % 360 # point of impact to tank hdg angle

        side = 2 if 135 < incoming < 225  else 1 if 45 < incoming < 315 else 0
        impact_angle = (p.forward_vec().angle_to(target.forward_vec()) + 360) % 360
        
        if side == 0:
            penetrated = 175 < impact_angle < 185
        elif side == 1:
            penetrated = 75 < impact_angle < 105 or 255 < impact_angle < 285
        else:
            penetrated = impact_angle < 60 or impact_angle > 300
        return penetrated



    def _line_of_fire_reward(self, shooter: Tank, target: Tank) -> float:
        diff = abs(((target.pos - shooter.pos).angle_to(shooter.turret_vec())) % 360)
        diff = min(diff, 360 - diff)
        aiming = diff < 5.5
        los = self._has_line_of_sight(shooter.pos, target.pos)
        # print(aiming if shooter == self.agent else None)
        if aiming and los:
            return 0.1
        elif aiming:
            return 0.02
        return -0.03

    def _has_line_of_sight(self, a: Vec2, b: Vec2) -> bool:
        steps = int((b - a).length() / 5)
        for i in range(1, steps):
            pt = a.lerp(b, i / steps)
            if any(w.collidepoint(pt.x, pt.y) for w in self.walls):
                return False
        return True



    def _get_obs(self) -> np.ndarray:
        agent_speed_vec = self.agent.forward_vec() * self.agent.MAX_SPEED
        enemy_speed_vec = self.enemy.forward_vec() * self.enemy.MAX_SPEED
        dx = self.agent.pos.x - self.enemy.pos.x
        dy = self.agent.pos.y - self.enemy.pos.y
        obs = np.array([
            dx / MapW,
            dy / MapH,
            min(self.agent.pos.x / MapW, 1 - self.agent.pos.x / MapW),
            min(self.agent.pos.y / MapH, 1 - self.agent.pos.y / MapH),
            (agent_speed_vec[0] -enemy_speed_vec[0]) / self.agent.MAX_SPEED / 2,
            (agent_speed_vec[1] - enemy_speed_vec[1]) / self.agent.MAX_SPEED / 2,
            math.cos(self.agent.hull_angle * _DEG2RAD - math.atan2(dy, dx)),
            math.sin(self.agent.hull_angle * _DEG2RAD - math.atan2(dy, dx)),
            math.cos(self.agent.turret_angle * _DEG2RAD - math.atan2(dy, dx)),
            math.sin(self.agent.turret_angle * _DEG2RAD - math.atan2(dy, dx)),
            math.cos(self.enemy.hull_angle * _DEG2RAD - math.atan2(0-dy, 0-dx)),
            math.sin(self.enemy.hull_angle * _DEG2RAD - math.atan2(0-dy, 0-dx)),
            math.cos(self.enemy.turret_angle * _DEG2RAD - math.atan2(0-dy, 0-dx)),
            math.sin(self.enemy.turret_angle * _DEG2RAD - math.atan2(0-dy, 0-dx)),
            self.agent.cooldown / 20.0,
            self.agent.hp / 5.0,
            self.enemy.hp / 5.0
        ], dtype=np.float32)
        obs_enemy = np.array([
            0 - dx / MapW,
            0 - dy / MapH,
            min(self.enemy.pos.x / MapW, 1 - self.enemy.pos.x / MapW),
            min(self.enemy.pos.y / MapH, 1 - self.enemy.pos.y / MapH),
            (enemy_speed_vec[0] - agent_speed_vec[0]) / self.enemy.MAX_SPEED / 2,
            (enemy_speed_vec[1] - agent_speed_vec[1]) / self.enemy.MAX_SPEED / 2,
            math.cos(self.enemy.hull_angle * _DEG2RAD - math.atan2(0-dy, 0-dx)),
            math.sin(self.enemy.hull_angle * _DEG2RAD - math.atan2(0-dy, 0-dx)),
            math.cos(self.enemy.turret_angle * _DEG2RAD - math.atan2(0-dy, 0-dx)),
            math.sin(self.enemy.turret_angle * _DEG2RAD - math.atan2(0-dy, 0-dx)),
            math.cos(self.agent.hull_angle * _DEG2RAD - math.atan2(dy, dx)),
            math.sin(self.agent.hull_angle * _DEG2RAD - math.atan2(dy, dx)),
            math.cos(self.agent.turret_angle * _DEG2RAD - math.atan2(dy, dx)),
            math.sin(self.agent.turret_angle * _DEG2RAD - math.atan2(dy, dx)),
            self.enemy.cooldown / 20.0,
            self.enemy.hp / 5.0,
            self.agent.hp / 5.0
        ], dtype=np.float32)
        return obs, obs_enemy


    def render(self):
        if self.window is None:
            self._init_render()

        # --- OS event pump: prevents window freeze ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.window.fill((34, 34, 34))
        for rect in self.walls:
            pygame.draw.rect(self.window, (150, 90, 90), rect)
        self._draw_tank(self.enemy, (200, 60, 60))
        self._draw_tank(self.agent, (60, 200, 60))
        for p in self.projectiles:
            pygame.draw.circle(self.window, (240, 240, 80), (int(p.pos.x), int(p.pos.y)), 3)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), (1, 0, 2))

    def _init_render(self):
        pygame.init()
        self.window = pygame.display.set_mode((MapW, MapH)) if self.render_mode == "human" else pygame.Surface((MapW, MapH))
        self.clock = pygame.time.Clock()

    def _draw_tank(self, tank: Tank, colour):
        size_px = int(tank.SIZE * 2)
        surf = pygame.Surface((size_px, size_px), pygame.SRCALPHA)
        # base hull
        pygame.draw.rect(surf, colour, pygame.Rect(0, 0, size_px, size_px))
        # front arrow
        tri = [
            (size_px * 0.9, size_px * 0.5),  # tip
            (size_px * 0.7, size_px * 0.35),
            (size_px * 0.7, size_px * 0.65),
        ]
        pygame.draw.polygon(surf, (255, 255, 0), tri)
        rot = pygame.transform.rotate(surf, -tank.hull_angle)
        rect = rot.get_rect(center=(tank.pos.x, tank.pos.y))
        self.window.blit(rot, rect)
        gun_end = tank.pos + tank.turret_vec() * (tank.SIZE + 10)
        pygame.draw.line(self.window, (240, 240, 240), (int(tank.pos.x), int(tank.pos.y)), (int(gun_end.x), int(gun_end.y)), 3)

    # ------------------------------------------------------------------
    # Static map geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _build_walls() -> List[pygame.Rect]:
        return [
            pygame.Rect(200, 480, 20, 120),
            pygame.Rect(600, 0, 20, 120),
            
            pygame.Rect(310, 330, 180, 20),
            pygame.Rect(310, 250, 180, 20),

            pygame.Rect(310, 210, 20, 120),
            pygame.Rect(490, 250, 20, 140),
        ]

    # ------------------------------------------------------------------
    # Naive placeholder adversary policy for testing
    # ------------------------------------------------------------------

    


# ----------------------------------------------------------------------
# Manual smoke test
# ----------------------------------------------------------------------


    
def _manual_main():
    env = TankBattleEnv(render_mode="human")
    env._reset_debug_mode()
    
    
    for i in range(5000):
        obs, obs_enemy, _ = env.reset()
        done = False
        if i % 10 == 0:
            print("Episode", i)
        steps = 0
        while not done:
            steps += 1
            action = env.NOOP
            enemy_action = env.NOOP
            obs, obs_enemy, reward, done, _ = env.step(action, enemy_action)
            # print(env.agent.hp, env.enemy.hp, reward)

    env.close()

if __name__ == "__main__":
    _manual_main()

