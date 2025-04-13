#################### Assignment 3: Reinforcement Learning #####################
############################## Ujjwal Sharma ##############################
############################## Roll No: 23M0837 ##############################
import highway_env
import gymnasium
from gymnasium.wrappers import RecordVideo
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cma
from cma.optimization_tools import EvalParallel2
import argparse
import warnings
warnings.filterwarnings('ignore')

env = gymnasium.make('racetrack-v0', render_mode='rgb_array')



def feature_extraction(state, info):

    # Feature one is the positon of the lane
    mean_point = 6
    rowCenteresWeighted = []
    rowCentered = []
    avg_center_point = 0
    avg_center_point_for_last_rows = 0
    avg_center_point_for_top_rows = 0


    speed = info["speed"]  # Current speed of the car
    
    dynamic_look_ahead_rows = 9 - speed // 4
    
    for row in range(len(state)):
        centerForRow = np.where(state[row] == 1)[0]
        centerPointForRow = 0
        if len(centerForRow) > 0:
            centerPointForRow = np.mean(centerForRow)
        # add the center points weighted according to the distance from the center and the row
        rowCenteresWeighted.append((row+1) * (centerPointForRow - mean_point))
        
        rowCentered.append(centerPointForRow-mean_point)
        
        avg_center_point += centerPointForRow
        
        if row >= 8:
            avg_center_point_for_last_rows += centerPointForRow
        else:
            avg_center_point_for_top_rows += centerPointForRow

    avg_center_point = avg_center_point / len(state)
    
    # avg_center_point_for_last_rows = avg_center_point_for_last_rows / (len(state) - 8)
    avg_center_point_for_last_rows = avg_center_point_for_last_rows / 5
    # 8 = avg_center_point_for_top_rows / 8
    avg_center_point_for_top_rows = avg_center_point_for_top_rows / 8
    # avg_center_point = np.sum(rowCenteres) / len(state)

    # steering_intensity = sum([1 if row>=9 else 0 for row in rowCenteresWeighted]) 

    # curvature detection
    # curvature = avg_center_point_for_last_rows - avg_center_point_for_top_rows
    
    
    # steering_intensity = np.sum([i*x if i>=9 else 0 for i,x in enumerate(rowCentered)])  # Steering intensity based on the distance from the center of the lane

    # Sensitivity factor based on speed
    sensitivity_factor = 1.0 - (speed / 20)

    steering_intensity = np.sum(rowCenteresWeighted[8:])/2  # Steering intensity based on the distance from the center of the lane
    
    speed = info["speed"]  # Current speed of the car

    # denominator_for_acceleration_intensity = (abs(mean_point - avg_center_point_for_last_rows) + 1)
    denominator_for_acceleration_intensity = (abs(mean_point - np.mean(rowCenteresWeighted[8:])/5 ) + 1)

    acceleration_intensity = sensitivity_factor *  (20 - speed) / denominator_for_acceleration_intensity # How far the car is from the center of the lane
    # acceleration_intensity = (20 - speed) / denominator_for_acceleration_intensity / steering_intensity # How far the car is from the center of the lane
    # for smaller speed the acceleration intensity is higher and positive 
    
        

    return rowCenteresWeighted , avg_center_point, avg_center_point_for_last_rows, steering_intensity, speed, acceleration_intensity


def policy(state, info, eval_mode = False, params = []):

    # The next 3 lines are used for reading policy parameters learned by training CMA-ES. Do not change them even if you don't use CMA-ES.
    if eval_mode:
        param_df = pd.read_json("cmaes_params.json")
        params = np.array(param_df.iloc[0]["Params"])

    """Replace the default policy given below by your policy"""

    # detect the centers of the lanes. the target position should be close to the center of the lanes and closer the center more the weightage
    # for the last lanes detect if the lane is left or right 

    rowCenteresWeighted , avg_center_point, avg_center_point_for_last_rows, steering_intensity, speed, acceleration_intensity = feature_extraction(state, info)

    steering_features = np.array(rowCenteresWeighted + [steering_intensity,avg_center_point_for_last_rows])  # Features for steering

    steering_unsqueezed = np.dot(steering_features, params[:15])  # Steering is a linear combination of features
    steering = np.tanh(steering_unsqueezed)  # Apply tanh activation function to get steering value

    # acceleration features are avg_center_point, speed, acceleration_intensity
    acceleration_features = np.array([avg_center_point_for_last_rows, avg_center_point, speed, acceleration_intensity])

    acceleration_unsqueezed = np.dot(acceleration_features, params[15:])  # Acceleration is a linear combination of features

    acceleration = np.tanh(acceleration_unsqueezed)  # Apply tanh activation function to get acceleration value
    

    return [acceleration, steering]



def fitness(params):
    total_distance = 0.0
    num_tracks = 6  # Evaluate on all 6 tracks
    for track in range(num_tracks):
        env.unwrapped.config["track"] = track
        (obs, info) = env.reset()
        state = obs[0]
        done = False
        while not done:
            action = policy(state, info, False, params)
            (obs, _, term, trunc, info) = env.step(action)
            state = obs[0]
            done = term or trunc
        total_distance += info["distance_covered"]
    avg_distance = total_distance / num_tracks
    return -avg_distance  # Minimize negative distance to maximize actual distance






def call_cma(num_gen=2, pop_size=2, num_policy_params = 1):
  sigma0 = 1
  x0 = np.random.normal(0, 1, (num_policy_params, 1))  # Initialisation of parameter vector
  opts = {'maxiter':num_gen, 'popsize':pop_size}
  es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
  with EvalParallel2(fitness, es.popsize + 1) as eval_all:
    while not es.stop():
      X = es.ask()
      es.tell(X, eval_all(X))
      es.logger.add()  # write data to disc for plotting
      es.disp()
  es.result_pretty()
  return es.result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')  # For training using CMA-ES
    parser.add_argument("--eval", action='store_true')  # For evaluating a trained CMA-ES policy
    parser.add_argument("--numTracks", type=int, default=6, required=False)  # Number of tracks for evaluation
    parser.add_argument("--seed", type=int, default=2025, required=False)  # Seed for evaluation
    parser.add_argument("--render", action='store_true')  # For rendering the evaluations
    args = parser.parse_args()

    train_mode = args.train
    eval_mode = args.eval
    num_tracks = args.numTracks
    seed = args.seed
    rendering = args.render

    """CMA-ES code begins"""
    # You can skip this part if you don't intend to use CMA-ES

    if train_mode:
        num_gen = 20
        pop_size = 20
        num_policy_params = 19
        X = call_cma(num_gen, pop_size, num_policy_params)
        cmaes_params = X[0]  # Parameters returned by CMA-ES after training
        cmaes_params_df = pd.DataFrame({
            'Params': [cmaes_params]
        })
        cmaes_params_df.to_json("cmaes_params.json")  # Storing parameters for evaluation purpose

    """CMA-ES code ends"""

    """Evaluation code begins"""
    # Do not modify this part.

    if rendering:
        env = RecordVideo(env, video_folder="videos", name_prefix="eval", episode_trigger=lambda x: True)

    if not train_mode:
        track_score_list = []  # This list stores the scores for different tracks

        for t in range(num_tracks):
            env.unwrapped.config["track"] = t  # Configuring the environment to provide track associated with index t. There are 6 tracks indexed 0 to 5.
            (obs, info) = env.reset(seed=seed)  # Getting initial state information from the environment
            state = obs[0]
            done = False

            while not done:  # While the episode is not done
                action = policy(state, info, eval_mode)  # Call policy to produce action
                (obs, _, term, trunc, info) = env.step(action)  # Take action in the environment
                state = obs[0]
                done = term or trunc  # If episode has terminated or truncated, set boolean variable done to True

            track_score = np.round(info["distance_covered"], 4).item()  # .item() converts numpy float to python float
            print("Track " + str(t) + " score:", track_score)
            track_score_list.append(track_score)

        env.close()

        # The next 4 lines of code generate a performance file which is used by autograder for evaluation. Don't change anything here.
        perf_df = pd.DataFrame()
        perf_df["Track_number"] = [n for n in range(num_tracks)]
        perf_df["Score"] = track_score_list
        perf_df.to_json("Performance_" + str(seed) + ".json")

        # A scatter plot is generated for you to visualise the performance of your agent across different tracks
        plt.scatter(np.arange(len(track_score_list)), track_score_list)
        plt.xlabel("Track index")
        plt.ylabel("Scores")
        plt.title("Scores across various tracks")
        plt.savefig('Evaluation.jpg')
        plt.close()

    """Code to generate learning curve and logs of CMA-ES"""
    # To be used only if your policy has parameters which are optimised using CMA-ES
    if train_mode:
        datContent = [i.strip().split() for i in open("outcmaes/fit.dat").readlines()]

        generations = []
        evaluations = []
        bestever = []
        best = []
        median = []
        worst = []

        for i in range(1, len(datContent)):
            generations.append(int(datContent[i][0]))
            evaluations.append(int(datContent[i][1]))
            bestever.append(-float(datContent[i][4]))
            best.append(-float(datContent[i][5]))
            median.append(-float(datContent[i][6]))
            worst.append(-float(datContent[i][7]))

        logs_df = pd.DataFrame()
        logs_df['Generations'] = generations
        logs_df['Evaluations'] = evaluations
        logs_df['BestEver'] = bestever
        logs_df['Best'] = best
        logs_df['Median'] = median
        logs_df['Worst'] = worst

        logs_df.to_csv('logs.csv')

        plt.plot(generations, best, color='green')
        plt.plot(generations, median, color='blue')
        plt.xlabel("Number of generations")
        plt.ylabel("Fitness")
        plt.legend(["Best", "Median"])
        plt.title('Evolution of fitness across generations')
        plt.savefig('LearningCurve.jpg')
        plt.close()

