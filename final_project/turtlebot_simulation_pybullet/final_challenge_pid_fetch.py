import pybullet as p
import time
import pybullet_data
import yaml
from cbs import cbs
import math
import threading


class PIDController:
    def __init__(self, kp, ki, kd) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative



def create_boundaries(length, width):
    """
        create rectangular boundaries with length and width

        Args:

        length: integer

        width: integer
    """
    for i in range(length):
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, -1, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, width, 0.5])
    for i in range(width):
        p.loadURDF("./final_challenge/assets/cube.urdf", [-1, i, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [length, i, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, -1, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, -1, 0.5])


def create_env(yaml_file):
    """
    Creates and loads assets only related to the environment such as boundaries and obstacles.
    Robots are not created in this function (check `create_turtlebot_actor`).
    """
    with open(yaml_file, 'r') as f:
        try:
            env_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e) 
            
    # Create env boundaries
    dimensions = env_params["map"]["dimensions"]
    create_boundaries(dimensions[0], dimensions[1])

    # Create env obstacles
    for obstacle in env_params["map"]["obstacles"]:
        p.loadURDF("./final_challenge/assets/cube.urdf", [obstacle[0], obstacle[1], 0.5])
    return env_params


def create_agents(yaml_file):
    """
    Creates and loads turtlebot agents.

    Returns list of agent IDs and dictionary of agent IDs mapped to each agent's goal.
    """
    agent_box_ids = []
    box_id_to_goal = {}
    agent_name_to_box_id = {}
    with open(yaml_file, 'r') as f:
        try:
            agent_yaml_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e)
    # if env_loaded:
    start_orientation = p.getQuaternionFromEuler([0,0,0])
    for agent in agent_yaml_params["agents"]:
        start_position = (agent["start"][0], agent["start"][1], 0)
        box_id = p.loadURDF("data/turtlebot.urdf", start_position, start_orientation, globalScaling=1)
        agent_box_ids.append(box_id)
        box_id_to_goal[box_id] = agent["goal"]
        agent_name_to_box_id[agent["name"]] = box_id
    return agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params


def read_cbs_output(file):
    """
        Read file from output.yaml, store path list.

        Args:

        output_yaml_file: output file from cbs.

        Returns:

        schedule: path to goal position for each robot.
    """
    with open(file, 'r') as f:
        try:
            params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return params["schedule"]


def checkPosWithBias(Pos, goal, bias):
    """
        Check if pos is at goal with bias

        Args:

        Pos: Position to be checked, [x, y]

        goal: goal position, [x, y]

        bias: bias allowed

        Returns:

        True if pos is at goal, False otherwise
    """
    if(Pos[0] < goal[0] + bias and Pos[0] > goal[0] - bias and Pos[1] < goal[1] + bias and Pos[1] > goal[1] - bias):
        return True
    else:
        return False


def navigation(agent, goal1, schedule1, goal2, schedule2):
    global finish
    global finish2
    global lock
    global lock2
    """
        Set velocity for robots to follow the path in the schedule.

        Args:

        agents: array containing the IDs for each agent

        schedule: dictionary with agent IDs as keys and the list of waypoints to the goal as values

        index: index of the current position in the path.

        Returns:

        Leftwheel and rightwheel velocity.
    """
    basePos = p.getBasePositionAndOrientation(agent)
    index = 0
    dis_th = 0.4
    prev_time = time.time()

    # linear_pid = PIDController(kp=25, ki=0.1, kd=0.2)
    # angular_pid = PIDController(kp=10, ki=0.1, kd=0.2)

    
    while(not checkPosWithBias(basePos[0], goal1, dis_th)):
        basePos = p.getBasePositionAndOrientation(agent)
        next = [schedule1[index]["x"], schedule1[index]["y"]]
        if(checkPosWithBias(basePos[0], next, dis_th)):
            index = index + 1
            with lock:
                finish[agent] = not finish[agent]
            while(True):
                with lock:
                    if len(set(finish.values())) == 1:
                        break
                p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                time.sleep(0.01)
        if(index == len(schedule1)):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break
        x = basePos[0][0]
        y = basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule1[index]["y"] - y), (schedule1[index]["x"] - x))

        if(Orientation < 0):
            Orientation = Orientation + 2 * math.pi
        if(goal_direction < 0):
            goal_direction = goal_direction + 2 * math.pi
        theta = goal_direction - Orientation

        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        current = [x, y]
        distance = math.dist(current, next)
        k1, k2, A = 25, 10, 20
        if agent == list(finish.keys())[0]:
            k1 = 15
        linear = k1 * math.cos(theta)
        angular = k2 * theta
        # linear = linear_pid.compute(math.cos(theta), dt)
        # angular = angular_pid.compute(theta, dt)

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        # time.sleep(0.001)
    print(agent, "here")
    drop_single_cube(agent)

    while(not checkPosWithBias(basePos[0], goal2, dis_th)):
        basePos = p.getBasePositionAndOrientation(agent)
        next = [schedule2[index]["x"], schedule2[index]["y"]]
        if(checkPosWithBias(basePos[0], next, dis_th)):
            index = index + 1
            with lock2:
                finish2[agent] = not finish2[agent]
            while(True):
                with lock2:
                    if len(set(finish2.values())) == 1:
                        break
                p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                time.sleep(0.01)
        if(index == len(schedule2)):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break
        x = basePos[0][0]
        y = basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule2[index]["y"] - y), (schedule2[index]["x"] - x))

        if(Orientation < 0):
            Orientation = Orientation + 2 * math.pi
        if(goal_direction < 0):
            goal_direction = goal_direction + 2 * math.pi
        theta = goal_direction - Orientation

        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        current = [x, y]
        distance = math.dist(current, next)
        k1, k2, A = 25, 10, 20
        if agent == list(finish.keys())[0]:
            k1 = 15
        linear = k1 * math.cos(theta)
        angular = k2 * theta
        # linear = linear_pid.compute(math.cos(theta), dt)
        # angular = angular_pid.compute(theta, dt)

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        # time.sleep(0.001)


def navigation2(agent, goal, schedule, goal2, schedule2):
    """
        Set velocity for robots to follow the path in the schedule.

        Args:

        agents: array containing the boxID for each agent

        schedule: dictionary with boxID as key and path to the goal as list for each robot.

        index: index of the current position in the path.

        Returns:

        Leftwheel and rightwheel velocity.
    """
    basePos = p.getBasePositionAndOrientation(agent)
    index = 0
    dis_th = 0.4
    while(not checkPosWithBias(basePos[0], goal, dis_th)):
        basePos = p.getBasePositionAndOrientation(agent)
        next = [schedule[index]["x"], schedule[index]["y"]]
        if(checkPosWithBias(basePos[0], next, dis_th)):
            index = index + 1
        if(index == len(schedule)):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break
        x = basePos[0][0]
        y = basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule[index]["y"] - y), (schedule[index]["x"] - x))
        if(Orientation < 0):
            Orientation = Orientation + 2 * math.pi
        if(goal_direction < 0):
            goal_direction = goal_direction + 2 * math.pi

        theta = goal_direction - Orientation

        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi

        current = [x, y]

        distance = math.dist(current, next)

        k1 = 20
        k2 = 5

        # linear = k1 * (distance) * math.cos(theta) + 5.0
        A=20
        # print(agent, "distance", distance)
        # print(agent, "exp", math.exp(k1 * distance))
        # print(agent, "distance", distance, "exp", math.exp(k1 * distance), A*math.exp(k1 * distance))
        # linear =min(A*math.exp(k1 * distance * math.cos(theta)), 24.0)
        linear = k1 * math.cos(theta)
        # print(agent, linear)
        angular = k2 * theta

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        # time.sleep(0.001)

    print(agent, "here")
    drop_single_cube(agent)
    basePos = p.getBasePositionAndOrientation(agent)

    index = 0
    while(not checkPosWithBias(basePos[0], goal2, dis_th)):
        basePos = p.getBasePositionAndOrientation(agent)
        next = [schedule2[index]["x"], schedule2[index]["y"]]
        if(checkPosWithBias(basePos[0], next, dis_th)):
            index = index + 1
        if(index == len(schedule2)):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break
        x = basePos[0][0]
        y = basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule2[index]["y"] - y), (schedule2[index]["x"] - x))
        if(Orientation < 0):
            Orientation = Orientation + 2 * math.pi
        if(goal_direction < 0):
            goal_direction = goal_direction + 2 * math.pi

        theta = goal_direction - Orientation

        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi

        current = [x, y]

        distance = math.dist(current, next)

        k1 = 20
        k2 = 5

        # linear = k1 * (distance) * math.cos(theta) + 5.0
        A=20
        # print(agent, "distance", distance)
        # print(agent, "exp", math.exp(k1 * distance))
        # print(agent, "distance", distance, "exp", math.exp(k1 * distance), A*math.exp(k1 * distance))
        # linear =min(A*math.exp(k1 * distance * math.cos(theta)), 24.0)
        linear = k1 * math.cos(theta)
        print(agent, linear, theta, angular)
        angular = k2 * theta

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        # time.sleep(0.001)




def read_input(yaml_file, env_loaded):
    """
        Read input file, load boundaries, robot and obstacles, set up goals dictionary

        Args:

        yaml_file: input yaml file

        env_loaded: True or false, check if the boundaries, robots and obstacles have been loaded before

        Returns:

        agents: list of boxID
        goals: dictionary of goal position for each robot.
        env_loaded: True
    """
    agents = []
    goals = {}

    with open(yaml_file, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
        if(env_loaded is True):
            for i in param["agents"]:
                goals[i["name"]] = i["goal"]
            return None, goals, True
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # disable tinyrenderer, software (CPU) renderer, we don't use it here
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        for i in param["agents"]:
            startPosition = (i["start"][0], i["start"][1], 0)
            boxId = p.loadURDF("data/turtlebot.urdf", startPosition, startOrientation, globalScaling=1)
            agents.append(boxId)
            goals[boxId] = i["goal"]
        dimensions = param["map"]["dimensions"]
        p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                     cameraTargetPosition=[7.5, 2.5, 0])

        create_boundaries(dimensions[0], dimensions[1])
        if env_loaded is False:
            create_env(yaml_file)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return agents, goals, True



def drop_single_cube(agent):
    """
        Drop cubes for each robot at their current positions.

        Args:

        agents: array containing the boxID for each agent
    """
    pos = p.getBasePositionAndOrientation(agent)[0]
    cube_pos = [pos[0], pos[1], 1]
    cubeID = p.loadURDF("data/small_cube.urdf", cube_pos, globalScaling=1)
    jointIndex = -1  # -1 implies the base
    parentFramePosition = [0, 0, 0.45]
    childFramePosition = [0, 0, 0]
    parentFrameOrientation = [0, 0, 0, 1]
    childFrameOrientation = [0, 0, 0, 1]

    fixedJoint = p.createConstraint(
        parentBodyUniqueId=agent,
        parentLinkIndex=jointIndex,
        childBodyUniqueId=cubeID,
        childLinkIndex=jointIndex,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=parentFramePosition,
        childFramePosition=childFramePosition,
        parentFrameOrientation=parentFrameOrientation,
        childFrameOrientation=childFrameOrientation
    )


def run(agents, goals1, schedule1, goals2, schedule2):
    """
        Set up loop to publish leftwheel and rightwheel velocity for each robot to reach goal position.

        Args:

        agents: array containing the boxID for each agent

        schedule: dictionary with boxID as key and path to the goal as list for each robot.

        goals: dictionary with boxID as the key and the corresponding goal positions as values
    """
    threads = []
    for agent in agents:
        t = threading.Thread(target=navigation2, args=(agent, goals1[agent], schedule1[agent], goals2[agent], schedule2[agent]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


# physics_client = p.connect(p.GUI, options='--width=1920 --height=1080 --mp4=multi_3.mp4 --mp4fps=10')
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# Disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

plane_id = p.loadURDF("plane.urdf")
startOrientation = p.getQuaternionFromEuler([0,0,0])
global env_loaded
env_loaded = False

# Create environment
env_params = create_env("./final_challenge/env.yaml")

# env_params2 = create_env("./final_challenge/env.yaml")
# Create turtlebots
# agent_box_ids1, agent_name_to_box_id1, box_id_to_goal1, agent_yaml_params1 = create_agents("./final_challenge/actors1.yaml")
# agent_box_ids2, agent_name_to_box_id2, box_id_to_goal2, agent_yaml_params2 = create_agents("./final_challenge/actors2.yaml")
agents, goals, env_loaded = read_input("final_challenge/env_2.yaml", env_loaded)
_,goals2,env_loaded = read_input("final_challenge/env_2_stage.yaml", env_loaded)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                     cameraTargetPosition=[4.5, 4.5, 4])


# cbs.run(dimensions=env_params["map"]["dimensions"], obstacles=env_params["map"]["obstacles"], agents=agent_yaml_params1["agents"], out_file="./final_challenge/cbs_output_fetch_1.yaml")
cbs_schedule1 = read_cbs_output("./final_challenge/cbs_output_fetch_1.yaml")
# Replace agent name with box id in cbs_schedule
# box_id_to_schedule_1 = {}
# for name, value in cbs_schedule1.items():
#     box_id_to_schedule_1[agent_name_to_box_id1[name]] = value
    
# cbs.run(dimensions=env_params["map"]["dimensions"], obstacles=env_params["map"]["obstacles"], agents=agent_yaml_params2["agents"], out_file="./final_challenge/cbs_output_fetch_2.yaml")
cbs_schedule2 = read_cbs_output("./final_challenge/cbs_output_fetch_2.yaml")
# box_id_to_schedule_2 = {}
# for name, value in cbs_schedule2.items():
#     box_id_to_schedule_2[agent_name_to_box_id2[name]] = value
    
# lock = threading.Lock()
# finish = {agent_box_ids1[0]: True, agent_box_ids1[1]: True}

# lock2 = threading.Lock()
# finish2 = {agent_box_ids2[0]: True, agent_box_ids2[1]: True}

# run(agent_box_ids1, box_id_to_goal1, box_id_to_schedule_1, box_id_to_goal2, box_id_to_schedule_2)
run(agents, goals, cbs_schedule1, goals2, cbs_schedule2)
time.sleep(2)
