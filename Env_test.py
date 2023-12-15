import cv2
from VesselEnv import VesselEnv, create_configs_rand
from gym_ras.env.wrapper import Visualizer


configs=create_configs_rand(1)
print(configs)
env = VesselEnv(configs=configs)
env = Visualizer(env)
obs = env.reset()
print("obs:", obs)
# action = env.action_space.sample()
# print(action)
    

done = False
while not done:
    while True:
        img = env.image
        img = cv2.resize(img, (640, 640))
        cv2.imshow("vessel", img)
        k = cv2.waitKey(0)
        cv2.setWindowTitle(
            'vessel', 'press n to continue, q to quit')
        if k & 0xFF == ord('q'):    # Esc key to stop
            done = True
            break
        elif k & 0xFF == ord('a'):
            action = 1
            obs, done = env.step(action)
            print(action, env.actions[action], done)
            break           
        elif k & 0xFF == ord('d'):
            action = 2
            obs, done = env.step(action)
            print(action, env.actions[action], done)
            break
        elif k & 0xFF == ord('w'):
            action = 3
            obs, done = env.step(action)
            print(action, env.actions[action], done)
            break
        elif k & 0xFF == ord('s'):
            action = 4
            obs, done = env.step(action)            
            print(action, env.actions[action], done)
            break
        elif k & 0xFF == ord('n'):
            action = 5
            obs, done = env.step(action)
            print(action, env.actions[action], done)
            break
        elif k & 0xFF == ord('c'):
            action = 6
            obs, done = env.step(action)
            print(action, env.actions[action], done)
            break
        else:
            continue
