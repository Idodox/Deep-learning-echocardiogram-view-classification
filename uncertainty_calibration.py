from run_state import Run


hyper_params = {"max_frames": 10
               ,"dataset": "10frame_5steps_100px"
               ,"classes": ['apex', 'papillary', 'mitral', '2CH', '3CH', '4CH']
               ,"model_type": "3dCNN"
               ,"resolution": 100
               ,"adaptive_pool": [7,5,5]
               ,"features": [16,16,"M",32,32,"M",32,32,"M",64,64,64,"M"]
               ,"classifier": [0.5,200,0.5,150,0.4,100]
                        }

checkpoint_path = '/Users/idofarhi/Documents/Thesis/Code/model_best.pt.tar'



run = Run(machine='server', hyper_params=hyper_params, inference=True, checkpoint_path=checkpoint_path)
run.model.eval()

with open(path, 'rb') as handle:
    file = pickle.load(handle)
