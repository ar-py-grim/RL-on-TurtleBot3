#!/dl_env/bin/env python3

import collections
import datetime
import json
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import sys
import threading
import time
import numpy
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn
import tensorflow as tf
from keras.layers import Dense, Input
from keras.losses import MeanSquaredError
from keras.models import load_model, Sequential
from keras.optimizers import Adam

LOGGING = False
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Memory growth enabled for {len(gpus)} GPU(s)")


class DQNMetric(tf.keras.metrics.Metric):

    def __init__(self, name='dqn_metric'):
        super(DQNMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')

    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)
        self.episode_step.assign_add(1)

    def result(self):
        return self.loss / self.episode_step

    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)


class DDQNAgent(Node):

    def __init__(self, stage_num, max_training_episodes, load_episode=0):
        super().__init__('ddqn_agent')

        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 26
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)

        self.done = False
        self.succeed = False
        self.fail = False

        # Hyperparameters used for the reported DDQN results
        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 20000 * self.stage
        self.epsilon_min = 0.05
        self.batch_size = 64

        self.replay_memory = collections.deque(maxlen=50000)
        self.min_replay_memory_size = 5000

        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.update_target_model()
        self.update_target_after = 5000
        self.target_update_after_counter = 0

        self.save_every = 20

        self.load_episode = int(load_episode)
        self.load_model = self.load_episode > 0
        self.model_dir_path = os.path.join(os.path.dirname(
                os.path.dirname(os.path.realpath(__file__))), 'model')
        os.makedirs(self.model_dir_path, exist_ok=True)
        self.model_path = os.path.join(
            self.model_dir_path, self.checkpoint_name(self.load_episode) + '.h5')

        if self.load_model:
            self.model.set_weights(load_model(self.model_path, compile=False).get_weights())
            # Sync the target network with the loaded weights, otherwise it stays
            # random until the first scheduled target update.
            self.update_target_model()
            with open(os.path.join(
                self.model_dir_path,
                self.checkpoint_name(self.load_episode) + '.json'
            )) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
                self.step_counter = param.get('step_counter')
            print(f'Loaded checkpoint: {self.model_path}')

        if LOGGING:
            tensorboard_file_name = current_time + ' ddqn_stage' + str(self.stage) + '_reward'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_dqn_logs', 'gradient_tape_ddqn', tensorboard_file_name
            )
            self.dqn_reward_writer = tf.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, 'get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

    def checkpoint_name(self, episode):
        return 'stage' + str(self.stage) + '_episode' + str(episode) + '_ddqn'

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = self.load_episode

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            if not rclpy.ok():
                break
            state = self.reset_environment()
            episode_num+= 1
            local_step = 0
            score = 0
            sum_max_q = 0.0

            time.sleep(1.0)

            while True:
                local_step+= 1

                # One forward pass, reused for the Q plot and for action selection
                q_values = self.model.predict(state, verbose=0)
                sum_max_q+= float(numpy.max(q_values))

                action = int(self.get_action(state, q_values))
                next_state, reward, done = self.step(action)
                score+= reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done)

                state = next_state

                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tf.summary.scalar(
                                'ddqn_reward', self.dqn_reward_metric.result(), step=episode_num
                            )
                        self.dqn_reward_metric.reset_states()

                    print(
                        'Episode:', episode,
                        'score:', score,
                        'memory length:', len(self.replay_memory),
                        'epsilon:', self.epsilon)
                    break

                time.sleep(0.01)

            if self.train_mode and episode % self.save_every == 0:
                self.save_model(episode)

    def save_model(self, episode):
        self.model_path = os.path.join(self.model_dir_path, self.checkpoint_name(episode) + '.h5')
        self.model.save(self.model_path)
        param_dictionary = {'epsilon': self.epsilon, 'step_counter': self.step_counter}
        with open(
            os.path.join(self.model_dir_path, self.checkpoint_name(episode) + '.json'), 'w') as outfile:
            json.dump(param_dictionary, outfile)
        print('Model saved:', self.model_path)

    def wait_for_future(self, future):
        # The MultiThreadedExecutor in main() is already spinning this node,
        # so the worker thread only has to wait for the response.
        while rclpy.ok() and not future.done():
            time.sleep(0.001)
        return future.result()

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...')

        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...')

        future = self.reset_environment_client.call_async(Dqn.Request())
        result = self.wait_for_future(future)

        if result is not None:
            state = numpy.reshape(numpy.asarray(result.state), [1, self.state_size])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))
            state = numpy.zeros([1, self.state_size])

        return state

    def get_action(self, state, q_values=None):
        if q_values is None:
            q_values = self.model.predict(state, verbose=0)

        if self.train_mode:
            self.step_counter+= 1
            self.epsilon = self.epsilon_min + (1.0-self.epsilon_min) * math.exp(-1.0*self.step_counter/self.epsilon_decay)
            lucky = random.random()
            if lucky > (1-self.epsilon):
                result = random.randint(0, self.action_size-1)
            else:
                result = numpy.argmax(q_values)
        else:
            result = numpy.argmax(q_values)

        return result

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)
        result = self.wait_for_future(future)

        if result is not None:
            next_state = numpy.reshape(numpy.asarray(result.state), [1, self.state_size])
            reward = result.reward
            done = result.done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))
            next_state = numpy.zeros([1, self.state_size])
            reward = 0.0
            done = True

        return next_state, reward, done

    def create_qnetwork(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        print('*Target model updated*')

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        data_in_mini_batch = random.sample(self.replay_memory, self.batch_size)

        current_states = numpy.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_qvalues_list = self.model.predict(current_states, verbose=0)

        next_states = numpy.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        # DDQN: the online network selects the next action, the target network evaluates it.
        next_qvalues_online = self.model.predict(next_states, verbose=0)
        next_qvalues_target = self.target_model.predict(next_states, verbose=0)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, _, done) in enumerate(data_in_mini_batch):
            current_q_values = current_qvalues_list[index]

            if not done:
                best_next_action = numpy.argmax(next_qvalues_online[index])
                future_reward = next_qvalues_target[index][best_next_action]
                desired_q = reward + self.discount_factor * future_reward
            else:
                desired_q = reward

            current_q_values[action] = desired_q
            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)
        x_train = numpy.reshape(x_train, [len(data_in_mini_batch), self.state_size])
        y_train = numpy.reshape(y_train, [len(data_in_mini_batch), self.action_size])

        self.model.fit(
            tf.convert_to_tensor(x_train, tf.float32),
            tf.convert_to_tensor(y_train, tf.float32),
            batch_size=self.batch_size, verbose=0
        )
        self.target_update_after_counter+= 1

        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()


def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    load_episode = args[3] if len(args) > 3 else '0'
    rclpy.init(args=args)
    ddqn_agent = DDQNAgent(stage_num, max_training_episodes, load_episode)

    executor = MultiThreadedExecutor()
    executor.add_node(ddqn_agent)
    process_thread = threading.Thread(target=ddqn_agent.process, daemon=True)
    process_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ddqn_agent.destroy_node()
        rclpy.shutdown()
        process_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
