# Import itertools
import itertools
# Define robots and their capabilities
robots = {
    "Forklift": ["L", "P"],
    "Drone": ["T"],
    "Conveyor": ["P"],
}

# Define tasks
tasks = {
    "Task 1": ["L"],
    "Task 2": ["T"],
    "Task 3": ["P"],
    "Task 4": ["L", "T"],  # Requires both lifting and transporting
}

# Deductive reasoning to assign tasks to robots (including cooperation)
task_assignments = {}

for task, requirements in tasks.items():
    suitable_robots = []
    
    # Check each robot's capabilities and match them to task requirements
    for robot, capabilities in robots.items():
        meets_requirements = all(req in capabilities for req in requirements)
        
        # Apply deductive reasoning
        if meets_requirements:
            suitable_robots.append(robot)
    
    # Check cooperative robot assignments
    if not suitable_robots:
        for combo in itertools.combinations(robots.keys(), len(requirements)):
            combo_capabilities = [cap for robot in combo for cap in robots[robot]]
            meets_requirements = all(req in combo_capabilities for req in requirements)
            
            if meets_requirements:
                suitable_robots.extend(combo)
                break
    
    # Assign suitable robots to the task
    task_assignments[task] = suitable_robots

# Display task assignments
for task, assigned_robots in task_assignments.items():
    if assigned_robots:
        print(f"{task} can be performed by: {', '.join(assigned_robots)}")
    else:
        print(f"No suitable robot found for {task}.")
