# 待修改
在视野内死亡
优化探索
优化timespans
meas增加探索坐标
stop simple mix

# 部分修改
base_class: 450行以下内容注释.  
configs: 注册部分游戏.  

# ENV
v21: DFP环境.  
  goal: 目标mea

  woods: 智能体炸掉多少 wood.  
  items: 智能体吃掉多少 item.  
  idx: 智能体编号. 10 ~ 13  
  my_bomb: 智能体炸弹位置. [[pos_x, pos_y, bomb_life, bomb_strength]]  
  ammo: 弹药量.  
  ammo_used: 目前使用了多少弹药.  
  frags: 敌人伤亡.  
  is_dead: 智能体是否阵亡.  
maze_v1:  
  reset->make_board  
       ->make_items  

# features
goal_map: 11 * 11 * 3  
imgs: 11 * 11 * 10  
goals and meas: 7dim: [woods↑, items↑, ammo_used↑, frags↑, is_dead↑, reach_goals↑, imove_counts↑]
scas: [steps, ammo, strength, kick, teammate, enemy1, enemy2]  

# run
``` python run.py --log_path=log/ --save_path=model/test --save_interval=1e5 --num_timesteps=1e7 ```

# 安装环境依赖包
可以 conda 初始化一个纯净环境，使用清华源或者豆瓣源安装   
```pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt```


# 后台运行
```nohup python -u *.py > logs/filename 2>&1 &```   


# pommerman 环境信息
{   
'alive': [10, 11],     
'board':    
array([[ 0,  0,  1,  7,  0,  0,  1,  2],  
       [ 0,  0,  0,  0,  1,  1,  1,  2],  
       [ 1, 10,  0,  0,  2,  0,  0,  0],  
       [ 2,  0,  3,  0,  1,  2,  1,  2],  
       [ 0,  1,  2,  1,  0,  1,  0,  0],  
       [ 0,  1,  0,  2,  1,  0,  0,  2],  
       [ 1,  1,  0,  1, 11,  0,  0,  0],  
       [ 2,  2,  0,  2,  3,  2,  0,  0]], dtype=uint8),    
'bomb_blast_strength':    
array([[0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 2., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 2., 0., 0., 0.]]),    
'bomb_life':    
array([[0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 7., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 8., 0., 0., 0.]]),    
'bomb_moving_direction':    
array([[0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.]]),     
'flame_life':     
array([[0., 0., 0., 0., 0., 0., 0., 0.],    
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.],   
       [0., 0., 0., 0., 0., 0., 0., 0.]]),    
'game_type': 4, 'game_env': 'pommerman.envs.v0:Pomme', 'position': (2, 1),    
'blast_strength': 2, 'can_kick': False, 'ammo': 0,    
'teammate': <Item.AgentDummy: 9>, 'enemies': [<Item.Agent1: 11>], 'step_count': 14}   
