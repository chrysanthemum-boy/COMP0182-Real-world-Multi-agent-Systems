# COMP0182-Real-world-Multi-agent-Systems


### Run NMPC
```bash
git clone https://github.com/chrysanthemum-boy/COMP0182-Real-world-Multi-agent-Systems.git
cd COMP0182-Real-world-Multi-agent-Systems
cd final_project
python nmpc_multi.py
```

### Run RRT or RRT* or A*

```bash
cd COMP0182-Real-world-Multi-agent-Systems
cd turtlebot_simulation_pybullet
python multi_robot_navigation_final_challenge.py
```
Change the algorithm in the code `multi_robot_navigation_final_challenge.py`

A*
```python
cbs.main("final_challenge/env_2.yaml", "cbs_output_fetch_1.yaml",1)
schedule = read_output("cbs_output_fetch_1.yaml")
_,goals2,env_loaded = read_input("final_challenge/env_2_stage.yaml", env_loaded)
cbs.main("final_challenge/env_2_stage.yaml", "cbs_output_fetch_2.yaml",2)
schedule2 = read_output("cbs_output_fetch_2.yaml")
run(agents, goals, schedule, goals2, schedule2)
time.sleep(2)
```

RRT
```python
cbs_rrt.main("final_challenge/env_2.yaml", "cbs_output_fetch_1.yaml",1)
schedule = read_output("cbs_output_fetch_1.yaml")
_,goals2,env_loaded = read_input("final_challenge/env_2_stage.yaml", env_loaded)
cbs_rrt.main("final_challenge/env_2_stage.yaml", "cbs_output_fetch_2.yaml",2)
schedule2 = read_output("cbs_output_fetch_2.yaml")
run(agents, goals, schedule, goals2, schedule2)
time.sleep(2)
```
RRT*
```python
cbs_rrt_star.main("final_challenge/env_2.yaml", "cbs_output_fetch_1.yaml",1)
schedule = read_output("cbs_output_fetch_1.yaml")
_,goals2,env_loaded = read_input("final_challenge/env_2_stage.yaml", env_loaded)
cbs_rrt_star.main("final_challenge/env_2_stage.yaml", "cbs_output_fetch_2.yaml",2)
schedule2 = read_output("cbs_output_fetch_2.yaml")
run(agents, goals, schedule, goals2, schedule2)
time.sleep(2)
```