#!/dl_env/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import time
import numpy
import rclpy
from rclpy.node import Node
from keras.models import load_model
from turtlebot3_msgs.srv import Dqn


class DDQNTest(Node):

    def __init__(self, stage, load_episode):
        super().__init__('ddqn_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)

        self.state_size = 26
        self.action_size = 5

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'model', f'stage{self.stage}_episode{self.load_episode}_ddqn.h5')
        self.model = load_model(model_path, compile=False)
        self.get_logger().info(f'Loaded model: {model_path}')

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.run_test()

    def get_action(self, state):
        state = numpy.asarray(state)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(numpy.argmax(q_values[0]))

    def run_test(self):
        episode = 0
        while rclpy.ok():
            episode+= 1
            done = False
            init = True
            score = 0
            local_step = 0
            next_state = []

            time.sleep(1.0)

            while not done:
                local_step+= 1
                # No state exists before the first call, so the first action is straight ahead
                action = 2 if local_step == 1 else self.get_action(next_state)

                req = Dqn.Request()
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn(
                        'rl_agent interface service not available, waiting again...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.done() and future.result() is not None:
                    next_state = future.result().state
                    reward = future.result().reward
                    done = future.result().done
                    score += reward
                    init = False
                else:
                    self.get_logger().error(f'Service call failure: {future.exception()}')

                time.sleep(0.01)

            self.get_logger().info(
                f'Episode: {episode}, steps: {local_step}, score: {score:.2f}')


def main(args=None):
    rclpy.init(args=args if args else sys.argv)
    stage = sys.argv[1] if len(sys.argv) > 1 else '1'
    load_episode = sys.argv[2] if len(sys.argv) > 2 else '600'
    node = DDQNTest(stage, load_episode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
